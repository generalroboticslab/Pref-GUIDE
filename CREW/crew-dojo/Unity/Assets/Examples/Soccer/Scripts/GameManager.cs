using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;
using Unity.Netcode;
using Nakama;
using Dojo;
using Dojo.Netcode;

namespace Examples.Soccer
{
    public class GameManager : NetworkBehaviour
    {
        [Tooltip("Maximum game duration in seconds")]
        [SerializeField]
        private float _gameEpisodeDuration = 120.0f;

        [SerializeField]
        private GameObject prefabBluePlayer;

        [SerializeField]
        private GameObject prefabPurplePlayer;

        [SerializeField]
        private GameObject prefabBall;

        [SerializeField]
        private GameObject environment;

        private readonly List<NetworkObject> purplePlayers = new();
        private readonly List<NetworkObject> bluePlayers = new();
        private NetworkObject soccerBall;

        private DojoConnection connection;
        private DojoTransport transport;

        private readonly NetworkVariable<bool> _gameRunning = new(false);
        private readonly NetworkVariable<int> _scoreBlue = new(0);
        private readonly NetworkVariable<int> _scorePurple = new(0);
        private readonly NetworkVariable<float> _remainingTime = new(0);

        public event Action OnGameStarted;
        public event Action OnGameEnded;
        public event Action<bool> OnGameScored;

        public bool IsGameRunning => _gameRunning.Value;
        public int TeamBlueScore => _scoreBlue.Value;
        public int TeamPurpleScore => _scorePurple.Value;
        public float RemainingTime => _remainingTime.Value;

        private Label _gameState;
        private Label _gameScore;

        private void Awake()
        {
            connection = FindObjectOfType<DojoConnection>();
            transport = FindObjectOfType<DojoTransport>();

            if (connection.IsServer)
            {
                connection.OnMatchPlayerJoined += OnPlayerJoinedOrRoleChanged;
                connection.OnRoleChanged += OnPlayerJoinedOrRoleChanged;
            }

            var root = GetComponentInChildren<UIDocument>().rootVisualElement;
            _gameScore = root.Q<Label>("GameScore");
            _gameState = root.Q<Label>("GameState");
        }

        private void Update()
        {
            var toRemove = purplePlayers.Where(player => player == null).ToList();
            toRemove.ForEach(obj => purplePlayers.Remove(obj));

            toRemove = bluePlayers.Where(player => player == null).ToList();
            toRemove.ForEach(obj => bluePlayers.Remove(obj));

            if (IsGameRunning)
            {
                var minutes = Math.Floor(RemainingTime / 60.0f);
                var seconds = RemainingTime - minutes * 60.0f;
                _gameState.text = $"{minutes}min {seconds:0f}s";
            }
            else
            {
                _gameState.text = "Waiting for players";
            }
            _gameScore.text = $"(Blue) {TeamBlueScore} : {TeamPurpleScore} (Purple)";

            if (IsServer)
            {
                _remainingTime.Value = Math.Max(0.0f, _remainingTime.Value - Time.deltaTime);
            }
        }

        private void OnPlayerJoinedOrRoleChanged(IUserPresence user)
        {
            if (connection.MatchClients.TryGetValue(user, out var role) && role == DojoNetworkRole.Player)
            {
                if (transport.GetNetcodeIDByUser(user, out var userID))
                {
                    SpawnPlayer(userID);
                }
            }
        }

        private void SpawnPlayer(ulong userID)
        {
            if (IsServer)
            {
                SpawnSoccerBall();
                if (purplePlayers.Count < 2 && bluePlayers.Count < 2)
                {
                    if (UnityEngine.Random.value < 0.5f)
                    {
                        SpawnBluePlayer(userID);
                    }
                    else
                    {
                        SpawnPurplePlayer(userID);
                    }
                }
                else if (purplePlayers.Count < 2)
                {
                    SpawnPurplePlayer(userID);
                }
                else if (bluePlayers.Count < 2)
                {
                    SpawnBluePlayer(userID);
                }
                CheckEpisodeStart();
            }
        }

        private void SpawnBluePlayer(ulong userID)
        {
            var posZ = bluePlayers.Count > 0 ? 1.0f : -1.0f;

            var player = Instantiate(prefabBluePlayer, new Vector3(-2.0f, 1.0f, posZ),
                        Quaternion.LookRotation(new(1.0f, 0.0f, 0.0f), Vector3.up), environment.transform);

            var netObj = player.GetComponent<NetworkObject>();
            netObj.SpawnAsPlayerObject(userID);

            bluePlayers.Add(netObj);
        }

        private void SpawnPurplePlayer(ulong userID)
        {
            var posZ = purplePlayers.Count > 0 ? 1.0f : -1.0f;

            var player = Instantiate(prefabPurplePlayer, new Vector3(2.0f, 1.0f, posZ),
                        Quaternion.LookRotation(new(-1.0f, 0.0f, 0.0f), Vector3.up), environment.transform);

            var netObj = player.GetComponent<NetworkObject>();
            netObj.SpawnAsPlayerObject(userID);

            purplePlayers.Add(netObj);
        }

        public void SpawnAIPlayer(AIAgent agent, bool isBlue)
        {
            if (IsServer)
            {
                SpawnSoccerBall();

                var prefab = isBlue ? prefabBluePlayer : prefabPurplePlayer;
                var posZ = isBlue ? (bluePlayers.Count > 0 ? 1.0f : -1.0f) : (purplePlayers.Count > 0 ? 1.0f : -1.0f);

                var player = Instantiate(prefab, new Vector3(isBlue ? -2.0f : 2.0f, 1.0f, posZ),
                            Quaternion.LookRotation(new(isBlue ? 1.0f : -1.0f, 0.0f, 0.0f), Vector3.up), environment.transform);

                var netObj = player.GetComponent<NetworkObject>();
                netObj.Spawn();

                var controller = player.GetComponent<PlayerController>();
                agent.SubscribeController(controller);

                if (isBlue)
                {
                    bluePlayers.Add(netObj);
                }
                else
                {
                    purplePlayers.Add(netObj);
                }
                CheckEpisodeStart();
            }
        }

        private void SpawnSoccerBall()
        {
            if (soccerBall == null)
            {
                var ball = Instantiate(prefabBall, Vector3.zero,
                    Quaternion.identity, environment.transform);
                var netObj = ball.GetComponent<NetworkObject>();
                netObj.Spawn();
                soccerBall = netObj;
            }
        }

        public void CheckEpisodeStart()
        {
            if (bluePlayers.Count == 2 && purplePlayers.Count == 2)
            {
                Invoke(nameof(StartEpisode), 0f);
            }
        }

        private void StartEpisode()
        {
            if (IsServer)
            {
                if (IsGameRunning)
                {
                    return;
                }

                _gameRunning.Value = true;

                CancelInvoke(nameof(StartEpisode));
                Invoke(nameof(StopEpisode), _gameEpisodeDuration);

                _remainingTime.Value = _gameEpisodeDuration;

                OnGameStarted?.Invoke();
            }
        }

        public void StopEpisode()
        {
            if (IsServer)
            {
                if (!IsGameRunning)
                {
                    return;
                }

                _gameRunning.Value = false;

                CancelInvoke(nameof(StopEpisode));
                OnGameEnded?.Invoke();

                ResetPositions();
                CheckEpisodeStart();

                _scoreBlue.Value = 0;
                _scorePurple.Value = 0;
            }
        }

        public void ResetPositions()
        {
            var rnd = UnityEngine.Random.Range(0, 1);
            bluePlayers.ForEach(player =>
            {
                ResetPosition(player, true, rnd);
                rnd = 1 - rnd;
            });
            rnd = UnityEngine.Random.Range(0, 1);
            purplePlayers.ForEach(player =>
            {
                ResetPosition(player, false, rnd);
                rnd = 1 - rnd;
            });

            soccerBall.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            var rigidBody = soccerBall.GetComponent<Rigidbody>();
            rigidBody.velocity = Vector3.zero;
        }

        private void ResetPosition(NetworkObject netObj, bool isBlue, int position)
        {
            var posZ = position > 0 ? 1.0f : -1.0f;
            netObj.transform.SetPositionAndRotation(
                new Vector3(isBlue ? -2.0f : 2.0f, 1.0f, posZ),
                Quaternion.LookRotation(new(isBlue ? 1.0f : -1.0f, 0.0f, 0.0f), Vector3.up)
            );

            var rigidBody = netObj.GetComponent<Rigidbody>();
            rigidBody.velocity = Vector3.zero;
        }

        public void HasScored(bool isBlue)
        {
            if (IsServer)
            {
                if (isBlue)
                {
                    _scoreBlue.Value++;
                }
                else
                {
                    _scorePurple.Value++;
                }

                OnGameScored?.Invoke(isBlue);
                ResetPositions();
            }
        }
    }
}