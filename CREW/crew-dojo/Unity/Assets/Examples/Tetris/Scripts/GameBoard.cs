using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements;
using Dojo;

namespace Examples.Tetris
{
    [RequireComponent(typeof(TetrisBoard))]
    public class GameBoard : MonoBehaviour
    {
        private const string LOGSCOPE = "GameBoard";

        [SerializeField]
        private InputActionAsset _playerActions;

        private InputActionMap _playerControl;

        [Tooltip("Object move down in every N seconds")]
        [SerializeField]
        private float _moveDownSpeed = 1f;

        [Tooltip("Request decision from agent (if in Agent Mode) every N seconds")]
        [SerializeField]
        private float _agentDecisionFrequency = 1.5f;

        [Tooltip("Send state update in every N seconds")]
        [SerializeField]
        private float _stateUpdateFreq = 0.1f;

        [SerializeField]
        private GameObject _tetrisUnitPrefab;

        [SerializeField]
        private AIAgentManager _aiAgentManager;

        [SerializeField]
        private UIDocument _scoreUI;

        private Label _scoreUILabel;

        [HideInInspector] public DojoConnection Connection;
        private bool IsClient => Connection.IsClient;
        private bool IsPlayer => Connection.IsPlayer;
        private bool IsClientConnected = false;

        private bool _autoTimestep = true;
        // in special mode, step down is requested by agent
        private bool _specialMode = false;

        [HideInInspector] public event Action<NetCommand> OnNewAction;
        [HideInInspector] public event Action<byte[]> OnNewState;
        [HideInInspector] public event Action OnGameOver;
        [HideInInspector] public event Action<int, int> OnScoreUpdate;
        [HideInInspector] public event Action OnNewObjectSpawned;

        private int _score = -1;
        public int Score
        {
            get
            {
                return _score;
            }
            private set
            {
                if (_scoreUI != null && _score != value)
                {
                    _score = value;
                    _scoreUILabel.text = $"Score: {_score}";
                }
            }
        }

        private TetrisBoard _board;
        private TetrisUnit[,] _states; // board states

        // controlled object positions
        private List<Vector2Int> _controlled = new();
        private readonly int _controlledCenter = 1;

        private void Awake()
        {
            _playerControl = _playerActions.actionMaps[0];
            _playerControl.Enable();
            _board = GetComponent<TetrisBoard>();

            var root = _scoreUI.rootVisualElement;
            _scoreUILabel = root.Q<Label>("Score");

            OnValidate();
        }

        private void OnValidate()
        {
            ValidateGameState();
        }

        private void Update()
        {
            HandleClientControl();
        }

        private void FixedUpdate()
        {
            if (!IsClient)
            {
                // check if player exists
                var playerCount = Connection.MatchClients.Values.Where(role => role != DojoNetworkRole.Viewer).Count() + Connection.MatchAIPlayers.Count;
                if ((playerCount > 0 && !IsClientConnected) || (playerCount == 0 && IsClientConnected))
                {
                    if (playerCount == 0 && IsClientConnected)
                    {
                        CallSpecialMode(false);
                    }
                    ResetGameState();
                }
                IsClientConnected = playerCount > 0;

                if (_specialMode && !IsInvoking(nameof(NextState)) && _controlled.Count == 0)
                {
                    // let object spawn
                    Invoke(nameof(NextState), 0.0f);
                }
            }
        }

        #region Controls

        // handle control in client mode
        private void HandleClientControl()
        {
            if (IsClient && IsPlayer)
            {
                if (_playerControl["Left"].WasPressedThisFrame())
                {
                    OnNewAction?.Invoke(NetCommand.MoveLeft);
                }
                if (_playerControl["Right"].WasPressedThisFrame())
                {
                    OnNewAction?.Invoke(NetCommand.MoveRight);
                }
                if (_playerControl["Rotate"].WasPressedThisFrame())
                {
                    OnNewAction?.Invoke(NetCommand.Rotate);
                }
                if (_playerControl["Flip"].WasPressedThisFrame())
                {
                    OnNewAction?.Invoke(NetCommand.Flip);
                }
                if (_playerControl["Drop"].WasPressedThisFrame())
                {
                    OnNewAction?.Invoke(NetCommand.Drop);
                }
            }
        }

        // handle control in server mode or AI mode
        public void HandleClientControl(NetCommand command)
        {
            if (IsClient && IsPlayer)
            {
                OnNewAction?.Invoke(command);
            }
            else if (!IsClient && _controlled.Count > 0)
            {
                if (command != NetCommand.Drop)
                {
                    var positions = _controlled;
                    var center = positions[_controlledCenter];
                    var newPositions = positions;
                    var color = _states[center.x, center.y].UnitColor;

                    if (command == NetCommand.MoveLeft)
                    {
                        newPositions = newPositions.Select(pos => new Vector2Int(pos.x - 1, pos.y)).ToList();
                    }
                    else if (command == NetCommand.MoveRight)
                    {
                        newPositions = newPositions.Select(pos => new Vector2Int(pos.x + 1, pos.y)).ToList();
                    }
                    else if (command == NetCommand.Rotate)
                    {
                        newPositions = newPositions.Select(pos =>
                        {
                            var offset = pos - center;
                            pos = center + new Vector2Int(
                                -offset.y,
                                +offset.x
                            );
                            return pos;
                        }).ToList();
                    }
                    else if (command == NetCommand.Flip)
                    {
                        newPositions = newPositions.Select(pos => new Vector2Int(center.x * 2 - pos.x, pos.y)).ToList();
                    }

                    SetStateAtPositions(positions, TetrisState.Unoccupied);
                    if (SetStateIfNotOccupied(newPositions, TetrisState.InControl))
                    {
                        _controlled = newPositions;
                        _controlled.ForEach(pos =>
                        {
                            _states[pos.x, pos.y].UnitColor = color;
                        });
                    }
                    else
                    {
                        SetStateAtPositions(positions, TetrisState.InControl);
                    }
                }
                else
                {
                    while (MoveDown()) ;
                }
            }
        }

        private bool MoveDown()
        {
            var positions = _controlled;
            var center = positions[_controlledCenter];
            var newPositions = positions.Select(pos => new Vector2Int(pos.x, pos.y + 1)).ToList();
            var color = _states[center.x, center.y].UnitColor;

            SetStateAtPositions(positions, TetrisState.Unoccupied);
            if (SetStateIfNotOccupied(newPositions, TetrisState.InControl))
            {
                _controlled = newPositions;
                _controlled.ForEach(pos =>
                {
                    _states[pos.x, pos.y].UnitColor = color;
                });
                return true;
            }
            else
            {
                SetStateAtPositions(positions, TetrisState.InControl);
                return false;
            }
        }

        #endregion Controls

        #region Timestep Control

        public void CallToggleTimestep()
        {
            _autoTimestep = !_autoTimestep;
            if (_autoTimestep && !IsInvoking(nameof(NextState)) && !_specialMode)
            {
                InvokeRepeating(nameof(NextState), _moveDownSpeed, _moveDownSpeed);
                InvokeRepeating(nameof(RequestAgentDecision), 0f, _agentDecisionFrequency);
            }
            else if (!_autoTimestep)
            {
                CancelInvoke(nameof(NextState));
                CancelInvoke(nameof(RequestAgentDecision));
            }
        }

        public void CallNextTimestep()
        {
            if (!_autoTimestep && !IsInvoking(nameof(NextState)))
            {
                NextState();
            }
        }

        public void CallSpecialMode(bool enable)
        {
            _specialMode = enable;
            if (_specialMode && IsInvoking(nameof(NextState)))
            {
                CancelInvoke(nameof(NextState));
            }
        }

        public void CallSpecialModeStepDown()
        {
            if (_specialMode)
            {
                Invoke(nameof(NextState), 0.0f);
            }
        }

        #endregion Timestep Control

        #region State Update

        public void HandleEvents(List<string> data)
        {
            var name = data[0];
            if (name.Equals("GameOver"))
            {
                OnGameOver?.Invoke();
            }
            else if (name.Equals("ScoreUpdate"))
            {
                if (data.Count >= 3 && int.TryParse(data[1], out var score) && int.TryParse(data[2], out var increase))
                {
                    OnScoreUpdate?.Invoke(score, increase);
                }
            }
            else if (name.Equals("ObjectSpawned"))
            {
                OnNewObjectSpawned?.Invoke();
            }
            else
            {
                Debug.LogWarning($"{LOGSCOPE}: Invalid event {name}");
            }
        }

        private void NextState()
        {
            if (!IsClient && IsClientConnected)
            {
                // update object
                if (_controlled.Count == 0)
                {
                    ObjectSpawn();
                    OnNewObjectSpawned?.Invoke();
                }
                else if (!MoveDown())
                {
                    ObjectFree();
                }

                var gameOver = false;
                var scoreIncrease = 0;

                // start row clear
                var boardSize = _board.BoardSize;
                var overMargin = _board.GameOverMargin - 2;
                Queue<Tuple<int, int>> rowsToRemove = new(); // (row start idx, row count)

                // scan for complete rows
                var stopScan = false;
                var scanRow = boardSize.y - 1;
                for (; scanRow >= overMargin && !stopScan; --scanRow)
                {
                    gameOver = scanRow <= overMargin;
                    stopScan = true;
                    var completeRow = true;
                    for (var scanCol = 0; scanCol < boardSize.x; ++scanCol)
                    {
                        if (_states[scanCol, scanRow].State != TetrisState.Free)
                        {
                            completeRow = false;
                        }
                        else
                        {
                            stopScan = false;
                        }
                    }
                    if (completeRow)
                    {
                        scoreIncrease++;
                        for (var scanCol = 0; scanCol < boardSize.x; ++scanCol)
                        {
                            _states[scanCol, scanRow].State = TetrisState.Unoccupied;
                        }
                        if (rowsToRemove.TryPeek(out var lastRow) && (lastRow.Item1 - lastRow.Item2) == scanRow)
                        {
                            rowsToRemove.Dequeue();
                            rowsToRemove.Enqueue(new(lastRow.Item1, lastRow.Item2 + 1));
                        }
                        else
                        {
                            rowsToRemove.Enqueue(new(scanRow, 1));
                        }
                    }
                }

                // display game over threshold
                _board.DisplayMargin = scanRow <= overMargin + 2;

                // increase score
                if (scoreIncrease > 0)
                {
                    Score += scoreIncrease;
                    OnScoreUpdate?.Invoke(Score, scoreIncrease);
                }

                // move down lines
                var offset = 0;
                while (rowsToRemove.Count > 0)
                {
                    var row = rowsToRemove.Dequeue();
                    var loopUntil = rowsToRemove.TryPeek(out var lastRow) ? lastRow.Item1 : scanRow;
                    offset += row.Item2;
                    for (var r = row.Item1 - row.Item2; r > loopUntil; --r)
                    {
                        for (var scanCol = 0; scanCol < boardSize.x; ++scanCol)
                        {
                            if (_states[scanCol, r].State == TetrisState.Free)
                            {
                                _states[scanCol, r + offset].State = TetrisState.Free;
                                _states[scanCol, r + offset].UnitColor = _states[scanCol, r].UnitColor;
                                _states[scanCol, r].State = TetrisState.Unoccupied;
                            }
                        }
                    }
                }

                // check game over
                if (gameOver)
                {
                    OnGameOver?.Invoke();
                    ResetGameState();
                }
            }
        }

        private void RequestAgentDecision()
        {
            AIAgent agent = _aiAgentManager.agents.FirstOrDefault();
            if (agent != null)
                agent.RequestDecision();
        }

        private void NextTick()
        {
            if (!IsClient && IsClientConnected)
            {
                var state = EncodeState();
                OnNewState?.Invoke(state);
            }
        }

        private void ValidateGameState()
        {
            if (_board == null)
            {
                return;
            }
            var size = _board.BoardSize;
            if (_states == null || _states.GetLength(0) != size.x || _states.GetLength(1) != size.y)
            {
                Score = 0;
                if (_states != null)
                {
                    for (var i = 0; i < _states.GetLength(0); ++i)
                    {
                        for (var j = 0; j < _states.GetLength(1); ++j)
                        {
                            ObjectDestroy(_states[i, j]);
                        }
                    }
                }
                _states = new TetrisUnit[size.x, size.y];
                for (var i = 0; i < size.x; ++i)
                {
                    for (var j = 0; j < size.y; ++j)
                    {
                        _states[i, j] = ObjectCreate();
                        _states[i, j].Board = _board;
                        _states[i, j].BoardPosition = new Vector2Int(i, j);
                        _states[i, j].State = TetrisState.Unoccupied;
                    }
                }
            }
        }

        public async void ResetGameState()
        {
            CancelInvoke(nameof(NextState));
            CancelInvoke(nameof(NextTick));
            CancelInvoke(nameof(RequestAgentDecision));

            Score = 0;
            var size = _board.BoardSize;
            for (var i = 0; i < size.x; ++i)
            {
                for (var j = 0; j < size.y; ++j)
                {
                    _states[i, j].State = TetrisState.Unoccupied;
                }
            }
            _board.DisplayMargin = false;
            _controlled.Clear();
            NextTick();

            await Task.Delay(1000);

            if (!IsClient)
            {
                if (_autoTimestep && !_specialMode)
                {
                    InvokeRepeating(nameof(NextState), _moveDownSpeed, _moveDownSpeed);
                    InvokeRepeating(nameof(RequestAgentDecision), 0f, _agentDecisionFrequency);
                }
                InvokeRepeating(nameof(NextTick), _stateUpdateFreq, _stateUpdateFreq);
            }
        }

        public void SetStateAtPositions(List<Vector2Int> positions, TetrisState state)
        {
            positions.ForEach(pos =>
            {
                if (IsPosValid(pos))
                {
                    _states[pos.x, pos.y].State = state;
                }
            });
        }

        public bool SetStateIfNotOccupied(List<Vector2Int> positions, TetrisState state)
        {
            if (positions.Any(pos => IsPosOccupied(pos)))
            {
                return false;
            }
            SetStateAtPositions(positions, state);
            return true;
        }

        public bool IsPosOccupied(Vector2Int pos)
        {
            if (!IsPosValid(pos))
            {
                return true;
            }
            return _states[pos.x, pos.y].State != TetrisState.Unoccupied;
        }

        public bool IsPosValid(Vector2Int pos)
        {
            return pos.x >= 0 && pos.x < _states.GetLength(0) && pos.y >= 0 && pos.y < _states.GetLength(1);
        }

        public async Task<Tuple<TetrisUnit[,], List<Vector2Int>>> GetInternalStates()
        {
            await Task.Delay((int)(1000 * _moveDownSpeed));
            while (_controlled.Count == 0 || IsInvoking(nameof(NextState)))
            {
                await Task.Delay(50);
            }
            return Tuple.Create(_states, _controlled);
        }

        #endregion State Update

        #region Game State Encoding

        // encode current state
        public byte[] EncodeState()
        {
            Debug.Assert(!IsClient);
            using var stream = new MemoryStream();
            using var writer = new BinaryWriter(stream);

            writer.Write(Score);
            writer.Write(_board.DisplayMargin);
            var size = _board.BoardSize;
            for (var i = 0; i < size.x; ++i)
            {
                for (var j = 0; j < size.y; ++j)
                {
                    var obj = _states[i, j];
                    writer.Write((sbyte)obj.State);
                    if (obj.State != TetrisState.Unoccupied)
                    {
                        var color = obj.UnitColor;
                        writer.Write(color.r);
                        writer.Write(color.g);
                        writer.Write(color.b);
                    }
                }
            }

            writer.Write(_controlled.Count);
            foreach (var pos in _controlled)
            {
                writer.Write(pos.x);
                writer.Write(pos.y);
            }

            return stream.ToArray();
        }

        // refer to https://www.cs.utexas.edu/~bradknox/papers/icdl08-knox.pdf
        public List<float> EncodeStateForUnityAgents()
        {
            Debug.Assert(IsClient);
            var features = new List<float>();
            var size = _board.BoardSize;

            // count heights and holes for each column
            var heights = new List<float>();
            var totalHoles = 0;
            for (var col = 0; col < size.x; ++col)
            {
                var lastHoles = 0;
                var maxHeight = 0;
                for (var row = size.y - 1; row >= 0; --row)
                {
                    var actualRow = size.y - row;
                    var obj = _states[col, row];
                    if (obj.State != TetrisState.Unoccupied)
                    {
                        totalHoles += lastHoles;
                        lastHoles = 0;
                        maxHeight = actualRow;
                    }
                    else
                    {
                        lastHoles++;
                    }
                }
                heights.Add(maxHeight);
            }

            // add features (21 values)
            features.AddRange(heights);
            features.Add(heights.Max());
            for (var idx = 1; idx < heights.Count; ++idx)
            {
                features.Add(Math.Abs(heights[idx] - heights[idx - 1]));
            }
            features.Add(totalHoles);

            return features;
        }

        public List<float> EncodeStateForUnityAgentsRaw()
        {
            Debug.Assert(IsClient);
            var features = new List<float>();

            var size = _board.BoardSize;
            features.Add(size.x);
            features.Add(size.y);

            for (var col = 0; col < size.x; ++col)
            {
                for (var row = size.y - 1; row >= 0; --row)
                {
                    var obj = _states[col, row];
                    features.Add((int)obj.State);
                }
            }

            return features;
        }

        // decode into current state
        public void DecodeState(byte[] data)
        {
            Debug.Assert(IsClient);

            using var stream = new MemoryStream(data);
            using var reader = new BinaryReader(stream);

            Score = reader.ReadInt32();
            _board.DisplayMargin = reader.ReadBoolean();
            var size = _board.BoardSize;
            for (var i = 0; i < size.x; ++i)
            {
                for (var j = 0; j < size.y; ++j)
                {
                    var obj = _states[i, j];
                    var state = (TetrisState)reader.ReadSByte();
                    obj.State = state;
                    if (obj.State != TetrisState.Unoccupied)
                    {
                        var color = new Color(
                            reader.ReadSingle(),
                            reader.ReadSingle(),
                            reader.ReadSingle()
                        );
                        obj.UnitColor = color;
                    }
                }
            }

            _controlled.Clear();
            var count = reader.ReadInt32();
            for (var idx = 0; idx < count; ++idx)
            {
                _controlled.Add(new Vector2Int(
                    reader.ReadInt32(),
                    reader.ReadInt32()
                ));
            }
        }

        #endregion Game State Encoding

        #region Object Management

        private void ObjectSpawn()
        {
            if (_controlled.Count > 0)
            {
                return;
            }

            // random type
            var type = (TetrisType)UnityEngine.Random.Range(0, (int)TetrisType.Count);

            // random color
            var color = UnityEngine.Random.ColorHSV(0.0f, 1.0f, 1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f);
            color.r = color.r * 0.2f + 0.8f;
            color.g = color.g * 0.2f + 0.8f;
            color.b = color.b * 0.2f + 0.8f;

            // position
            var center = new Vector2Int(_board.BoardSize.x / 2, 1);
            var valid = ObjectSpawn(type, center, color);
            for (var x = -2; x < 2 && !valid; ++x)
            {
                valid = ObjectSpawn(type, center + new Vector2Int(x, 0), color);
            }
            Debug.Assert(valid, "SpawnObjects error, should always find a valid position!");
        }

        private bool ObjectSpawn(TetrisType type, Vector2Int center, Color color)
        {
            var positions = Enumerable.Repeat(center, 4).ToList();
            switch (type)
            {
                case TetrisType.TYPE1:
                    positions[0] += new Vector2Int(-1, 0);
                    positions[2] += new Vector2Int(1, 0);
                    positions[3] += new Vector2Int(2, 0);
                    break;
                case TetrisType.TYPE2:
                    positions[0] += new Vector2Int(-1, 0);
                    positions[2] += new Vector2Int(0, -1);
                    positions[3] += new Vector2Int(1, 0);
                    break;
                case TetrisType.TYPE3:
                    positions[0] += new Vector2Int(0, -1);
                    positions[2] += new Vector2Int(1, 0);
                    positions[3] += new Vector2Int(2, 0);
                    break;
                case TetrisType.TYPE4:
                    positions[0] += new Vector2Int(0, -1);
                    positions[2] += new Vector2Int(1, 0);
                    positions[3] += new Vector2Int(1, -1);
                    break;
                case TetrisType.TYPE5:
                    positions[0] += new Vector2Int(-1, 0);
                    positions[2] += new Vector2Int(0, -1);
                    positions[3] += new Vector2Int(1, -1);
                    break;
            }
            if (SetStateIfNotOccupied(positions, TetrisState.InControl))
            {
                _controlled = positions;
                _controlled.ForEach(pos =>
                {
                    _states[pos.x, pos.y].UnitColor = color;
                });
                return true;
            }
            else
            {
                return false;
            }
        }

        private void ObjectFree()
        {
            if (_controlled.Count > 0)
            {
                SetStateAtPositions(_controlled, TetrisState.Free);
                _controlled.Clear();
            }
        }

        private TetrisUnit ObjectCreate()
        {
            return Instantiate(_tetrisUnitPrefab, transform).GetComponent<TetrisUnit>();
        }

        private void ObjectDestroy(TetrisUnit obj)
        {
            Destroy(obj.gameObject);
        }

        #endregion Object Management
    }
}