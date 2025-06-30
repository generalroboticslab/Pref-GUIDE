using System;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using Unity.MLAgents.Sensors;
using Dojo;
using System.Collections.Generic;

namespace Examples.Tetris
{
    public class AIAgentManager : MonoBehaviour
    {
        [SerializeField]
        private GameBoard _board;

        [SerializeField]
        private GameObject _agentPrefab;

        [SerializeField]
        private Camera _aiAgentCamera;

        private DojoConnection _connection;

        private EventChannel _eventChannel;
        private ToggleTimestepChannel _toggleTimestepChannel;

        private int _numAgents = 0;

        [HideInInspector]
        public readonly List<AIAgent> agents = new();

        private void Awake()
        {
#if UNITY_STANDALONE // && !UNITY_EDITOR

            var args = Environment.GetCommandLineArgs();

            for (var idx = 0; idx < args.Length; ++idx)
            {
                var arg = args[idx];

                if (arg.Equals("-NumAgents") && idx < args.Length - 1 && int.TryParse(args[idx + 1], out var num) && num >= 0)
                {
                    _numAgents = num;
                    ++idx;
                }
            }
#endif
            _connection = FindObjectOfType<DojoConnection>();
            _connection.OnJoinedMatch += OnJoinedMatch;

            CameraSensorComponent cameraSensorComponent = _agentPrefab.GetComponent<CameraSensorComponent>();
            cameraSensorComponent.Camera = _aiAgentCamera;
        }

        private void OnJoinedMatch()
        {
            if (_connection.IsServer)
            {
                if (_numAgents > 0)
                {
                    // register AI players
                    var players = Enumerable.Range(0, _numAgents).Select(x => $"Tetris-{x}").ToList();
                    _numAgents = _connection.RegisterAIPlayers(players);

                    // spawn AI players
                    for (var i = 0; i < _numAgents; ++i)
                    {
                        var agent = Instantiate(_agentPrefab, transform).GetComponent<AIAgent>();
                        agents.Add(agent);
                    }
                    Initialize();
                }
            }
        }

        private void Initialize()
        {
            Debug.Assert(_connection.IsServer);

            if (Academy.IsInitialized)
            {
                // register MLAgent environment
                _eventChannel = new(_connection, _board);
                if (_eventChannel.IsInitialized)
                {
                    SideChannelManager.RegisterSideChannel(_eventChannel);
                }
                _toggleTimestepChannel = new(_board);
                if (_toggleTimestepChannel.IsInitialized)
                {
                    SideChannelManager.RegisterSideChannel(_toggleTimestepChannel);
                }

                Academy.Instance.OnEnvironmentReset += _board.ResetGameState;
            }
        }

        private void OnDestroy()
        {
            if (Academy.IsInitialized)
            {
                if (_eventChannel.IsInitialized)
                {
                    SideChannelManager.UnregisterSideChannel(_eventChannel);
                }
                if (_toggleTimestepChannel.IsInitialized)
                {
                    SideChannelManager.UnregisterSideChannel(_toggleTimestepChannel);
                }
            }
        }
    }
}
