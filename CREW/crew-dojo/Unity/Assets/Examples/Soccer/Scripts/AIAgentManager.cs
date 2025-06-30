using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.SideChannels;
using Unity.MLAgents;
using Dojo;
using Unity.Netcode;

namespace Examples.Soccer
{
    public class AIAgentManager : MonoBehaviour
    {
        [SerializeField]
        private GameObject _blueAgentPrefab;

        [SerializeField]
        private GameObject _purpleAgentPrefab;

        [SerializeField]
        private GameManager _gameManager;

        private DojoConnection _connection;

        private int _numBlueAgents = 0;
        private int _numPurpleAgents = 0;

        private readonly List<AIAgent> _blueAgents = new();
        private readonly List<AIAgent> _purpleAgents = new();

        private EventChannel _eventChannel;

        private void Awake()
        {
#if UNITY_STANDALONE // && !UNITY_EDITOR

            var args = Environment.GetCommandLineArgs();

            for (var idx = 0; idx < args.Length; ++idx)
            {
                var arg = args[idx];

                if (arg.Equals("-NumBlueAgents") && idx < args.Length - 1 && int.TryParse(args[idx + 1], out var numBlue) && numBlue >= 0)
                {
                    _numBlueAgents = numBlue;
                    ++idx;
                }

                else if (arg.Equals("-NumPurpleAgents") && idx < args.Length - 1 && int.TryParse(args[idx + 1], out var numPurple) && numPurple >= 0)
                {
                    _numPurpleAgents = numPurple;
                    ++idx;
                }
            }
#endif

            _connection = FindObjectOfType<DojoConnection>();
        }

        private void Start()
        {
            if (_connection.IsServer)
            {
                RegisterAgents();

                NetworkManager.Singleton.OnServerStarted += OnServerStarted;
            }
        }

        public void ResetAgents()
        {
            _blueAgents.ForEach(agent => agent.RequestDecision());
            _purpleAgents.ForEach(agent => agent.RequestDecision());
        }

        public void EndEpisode()
        {
            if (_blueAgents.Count > 0)
            {
                _blueAgents.First().EndEpisode();
            }
            else if (_purpleAgents.Count > 0)
            {
                _purpleAgents.First().EndEpisode();
            }
        }

        private void RegisterAgents()
        {
            // register AI players
            if (_numBlueAgents > 0)
            {
                var players = Enumerable.Range(0, _numBlueAgents).Select(x => $"BlueAgent-{x}").ToList();
                _numBlueAgents = _connection.RegisterAIPlayers(players);
            }

            if (_numPurpleAgents > 0)
            {
                // Add _numBlueAgents to ensure all agent ids are unique
                var players = Enumerable.Range(0, _numPurpleAgents).Select(x => $"PurpleAgent-{x + _numBlueAgents}").ToList();
                _numPurpleAgents = _connection.RegisterAIPlayers(players) - _numBlueAgents;
            }

            for (var i = 0; i < _numBlueAgents; ++i)
            {
                var agent = Instantiate(_blueAgentPrefab, transform).GetComponent<AIAgent>();
                agent.AgentID = i;
                _blueAgents.Add(agent);
            }
            for (var i = 0; i < _numPurpleAgents; ++i)
            {
                var agent = Instantiate(_purpleAgentPrefab, transform).GetComponent<AIAgent>();
                agent.AgentID = i;
                _purpleAgents.Add(agent);
            }
            ConnectAgents();
        }

        private void ConnectAgents()
        {
            Debug.Assert(_connection.IsServer);

            // register MLAgent environment
            _eventChannel = new(_gameManager, _connection);
            if (_eventChannel.IsInitialized)
            {
                SideChannelManager.RegisterSideChannel(_eventChannel);
            }

            Academy.Instance.OnEnvironmentReset += _gameManager.StopEpisode;
            ResetAgents();
        }

        private void OnDestroy()
        {
            if (_eventChannel.IsInitialized)
            {
                SideChannelManager.UnregisterSideChannel(_eventChannel);
            }
        }

        private void OnServerStarted()
        {
            SpawnAgents();
        }

        public void SpawnAgents()
        {
            _blueAgents.ForEach(agent => _gameManager.SpawnAIPlayer(agent, true));
            _purpleAgents.ForEach(agent => _gameManager.SpawnAIPlayer(agent, false));
        }
    }
}