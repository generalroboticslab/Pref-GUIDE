using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using Unity.MLAgents.Sensors;
using Dojo;
using Unity.Netcode;
using Unity.Netcode.Components;

namespace Examples.FindTreasure
{
    public class AIAgentManager : MonoBehaviour
    {
        [SerializeField]
        private MapManager _map;

        [SerializeField]
        private GameObject _agentPrefab;

        [SerializeField]
        private Camera _aiAgentCamera;

        // [SerializeField]
        // private RenderTexture RT;

        // [SerializeField]
        // private AccumuCamera _accumuCamera;

        private DojoConnection _connection;

        private GameManager _gameManager;

        private PlayerController _playerController;

        private EventChannel _eventChannel;
        private WrittenFeedbackChannel _writtenFeedbackChannel;

        private Vector3 spawnpos;



        [HideInInspector]
        public AIAgent agent;

        private void Awake()
        {
            _connection = FindObjectOfType<DojoConnection>();
            _gameManager = FindObjectOfType<GameManager>();

            var cameraSensorComponent = _agentPrefab.GetComponent<CameraSensorComponent>();
            // _aiAgentCamera.enabled = true;
            // var renderSensorComponent = _agentPrefab.GetComponent<RenderTextureSensorComponent>();
            // renderSensorComponent.Camera = _aiAgentCamera;


            var cams = _agentPrefab.transform.Find("AccCam_Sensor").GetComponent<AccumuCamera>().GetComponent<Camera>();
            // Debug.Log($"{cams}");

            cameraSensorComponent.Camera = cams;
            // cameraSensorComponent.Camera = _aiAgentCamera;




            // CameraSensorComponent.RenderTextureSensorComponent = RT;
            // cameraSensorComponent.Camera = _aiAgentCamera;
            // Camera.main.depth = 10;



            // _playerController = _agentPrefab.GetComponentInChildren<PlayerController>();
            // _aiAgentCamera.enabled = true;


            // cameraSensorComponent.Camera = Camera.main;//_aiAgentCamera;
            // Debug.Log($"{_aiAgentCamera.enabled}");




            // Debug.Log($"{_playerController}");
            // cameraSensorComponent.Camera = _agentPrefab.transform.Find("AccCam").GetComponentInChildren<Camera>();
        }

        public void SpawnAgent()
        {
            if (!_connection.IsServer)
                throw new NotServerException("You must spawn agents on the server for server ownership");
            _connection.RegisterAIPlayers(new List<string> { "FindTreasure-0" });
            var netObj = Instantiate(_agentPrefab, _map.FindSpawnPointForPlayer().center, Quaternion.identity).GetComponent<NetworkObject>();
            agent = netObj.GetComponentInChildren<AIAgent>();
            agent.AgentID = 0;
            ResetAgent();
            netObj.Spawn();
            Initialize();

        }

        private void Initialize()
        {
            if (Academy.IsInitialized)
            {
                // register MLAgent environment
                _eventChannel = new(_connection);
                _writtenFeedbackChannel = new(_connection);
                if (_eventChannel.IsInitialized)
                    SideChannelManager.RegisterSideChannel(_eventChannel);
                if (_writtenFeedbackChannel.IsInitialized)
                    SideChannelManager.RegisterSideChannel(_writtenFeedbackChannel);

                Academy.Instance.OnEnvironmentReset += _gameManager.ResetGame;
            }
        }

        public void ResetAgent()
        {
            if (_connection.IsServer)
            {
                var spawnPoint = _map.FindSpawnPointForPlayer();
                Debug.Log($"Agent spawn point: {spawnPoint}");

                agent.GetComponentInChildren<PlayerController>().Teleport(spawnPoint.center);

                // var pos = new Vector3(-2, 0, 0);
                // agent.GetComponentInChildren<PlayerController>().Teleport(pos);






                // agent.transform.Find("Body").SetPositionAndRotation(spawnPoint.center, _agentPrefab.transform.localRotation);
                //agent.transform.Teleport(spawnPoint.center);
                agent.StartCoroutine(agent.WaitAndStartRequestingDecisions());
            }
            agent.GetComponentInChildren<PlayerController>().CamAcc.ClearAccumulation();
            agent.GetComponentInChildren<PlayerController>().CamAccSens.ClearAccumulation();
            agent.GetComponentInChildren<PlayerController>().clear_cam_flag.Value = true;
            agent.GetComponentInChildren<PlayerController>()._offset = Vector3.zero;
            agent.GetComponentInChildren<PlayerController>()._angleOffset = Vector3.zero;
        }


        private void OnDestroy()
        {
            if (Academy.IsInitialized)
            {
                if (_eventChannel.IsInitialized)
                    SideChannelManager.UnregisterSideChannel(_eventChannel);
                if (_writtenFeedbackChannel.IsInitialized)
                    SideChannelManager.UnregisterSideChannel(_writtenFeedbackChannel);
            }
        }



    }
}
