using System;
using System.Collections.Generic;
using UnityEngine;
using Dojo;
using Nakama.TinyJson;

namespace Examples.Tetris
{
    [DefaultExecutionOrder(-1)]
    public class GameManager : MonoBehaviour
    {
        [SerializeField]
        private GameBoard _board;

        [SerializeField]
        private DojoConnection _connection;

        private bool IsClient => _connection.IsClient;

        private void Awake()
        {
            Application.targetFrameRate = 60;
            Debug.Assert(FindObjectsOfType<GameManager>().Length == 1, "Only one game manager is allowed!");
        }

        private void Start()
        {
            // setup callbacks
            _board.Connection = _connection;
            _board.OnNewAction += NewClientAction;
            _board.OnNewState += NewServerState;

            _board.OnGameOver += NewGameOverEvent;
            _board.OnScoreUpdate += NewScoreUpdateEvent;
            _board.OnNewObjectSpawned += NewObjectSpawnedEvent;

            _connection.SubscribeRemoteMessages((long)NetOpCode.ClientAction, OnClientAction);
            _connection.SubscribeRemoteMessages((long)NetOpCode.ServerState, OnServerState);
            _connection.SubscribeRemoteMessages((long)NetOpCode.GameEvent, OnGameEvent);

            _connection.SubscribeRemoteMessages((long)NetOpCode.ToggleTimestep, OnToggleTimestep);
            _connection.SubscribeRemoteMessages((long)NetOpCode.NextTimestep, OnNextTimestep);

            _connection.SubscribeRemoteMessages((long)NetOpCode.SpecialModeRequest, OnSpecialModeRequest);
            _connection.SubscribeRemoteMessages((long)NetOpCode.SpecialModeStepDown, OnSpecialModeStepDown);
        }

        #region State Action Updates

        private void NewClientAction(NetCommand command)
        {
            if (IsClient)
            {
                var action = command.ToString();
                _connection.SendStateMessage((long)NetOpCode.ClientAction, action);
            }
        }

        private void NewServerState(byte[] state)
        {
            if (!IsClient)
            {
                _connection.SendStateMessage((long)NetOpCode.ServerState, state);
            }
        }

        private void OnClientAction(DojoMessage m)
        {
            if (!IsClient)
            {
                var action = m.GetString();
                if (Enum.TryParse(typeof(NetCommand), action, out var command))
                {
                    _board.HandleClientControl((NetCommand)command);
                }
                else
                {
                    Debug.LogWarning($"Invalid remote action: {action}");
                }
            }
        }

        private void OnServerState(DojoMessage m)
        {
            if (IsClient)
            {
                var state = m.RawData;
                _board.DecodeState(state);
            }
        }

        #endregion State Action Updates

        #region Timestep Controls

        private void OnToggleTimestep(DojoMessage m)
        {
            if (!IsClient)
            {
                _board.CallToggleTimestep();
            }
        }

        private void OnNextTimestep(DojoMessage m)
        {
            if (!IsClient)
            {
                _board.CallNextTimestep();
            }
        }

        private void OnSpecialModeRequest(DojoMessage m)
        {
            if (!IsClient)
            {
                _board.CallSpecialMode(true);
            }
        }

        private void OnSpecialModeStepDown(DojoMessage m)
        {
            if (!IsClient)
            {
                _board.CallSpecialModeStepDown();
            }
        }

        #endregion Timestep Controls

        #region Events

        private void NewGameOverEvent()
        {
            if (!IsClient)
            {
                var message = new List<string>() { "GameOver" };
                _connection.SendStateMessage((long)NetOpCode.GameEvent, JsonWriter.ToJson(message));
            }
        }

        private void NewScoreUpdateEvent(int score, int increase)
        {
            if (!IsClient)
            {
                var message = new List<string>() { "ScoreUpdate", score.ToString(), increase.ToString() };
                _connection.SendStateMessage((long)NetOpCode.GameEvent, JsonWriter.ToJson(message));
            }
        }

        private void NewObjectSpawnedEvent()
        {
            if (!IsClient)
            {
                var message = new List<string>() { "ObjectSpawned" };
                _connection.SendStateMessage((long)NetOpCode.GameEvent, JsonWriter.ToJson(message));
            }
        }

        private void OnGameEvent(DojoMessage m)
        {
            if (IsClient)
            {
                _board.HandleEvents(m.GetDecodedData<List<string>>());
            }
        }

        #endregion Events
    }
}