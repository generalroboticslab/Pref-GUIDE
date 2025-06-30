using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using Unity.MLAgents.Sensors;
using Dojo;
using Unity.Netcode;


namespace Examples.WallJump
{

    public class AIAgentManager : MonoBehaviour
    {
        // [SerializeField]
        // private MapManager _map;

        [SerializeField]
        private GameObject _agentPrefab;

        Bounds m_SpawnAreaBounds;

        public GameObject spawnArea;

        [SerializeField]
        private Camera _aiAgentCamera;

        [SerializeField]
        private PlayerController _playerController;

        private DojoConnection _connection;

        private GameManager _gameManager;


        private EventChannel _eventChannel;
        private WrittenFeedbackChannel _writtenFeedbackChannel;

        [HideInInspector]
        public AIAgent agent;

        
        // EnvironmentParameters m_ResetParams;

        private void Awake()
        {
            _connection = FindObjectOfType<DojoConnection>();
            _gameManager = FindObjectOfType<GameManager>();

            var cameraSensorComponent = _agentPrefab.GetComponent<CameraSensorComponent>();
            cameraSensorComponent.Camera = _aiAgentCamera;

            m_SpawnAreaBounds = spawnArea.GetComponent<Collider>().bounds;
            spawnArea.SetActive(false);
            // m_ResetParams = Academy.Instance.EnvironmentParameters;
            _playerController = GetComponentInChildren<PlayerController>();
        }

        public void SpawnAgent()
        {
            if (!_connection.IsServer)
                throw new NotServerException("You must spawn agents on the server for server ownership");
            _connection.RegisterAIPlayers(new List<string> { "WallJump-0" });
            var netObj = Instantiate(_agentPrefab).GetComponent<NetworkObject>();
            agent = netObj.GetComponentInChildren<AIAgent>();
            agent.AgentID = 0;
            ResetBlock();
            ResetWall();
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

                agent.transform.Find("GreenPlayer").SetPositionAndRotation(new Vector3(18 * (Random.value - 0.5f), 1, -9), _agentPrefab.transform.localRotation);
                agent.StartRequestingDecisions();
            
            }
        }

        public Vector3 GetRandomSpawnPos()
        {
            var randomPosX = Random.Range(-m_SpawnAreaBounds.extents.x,
                m_SpawnAreaBounds.extents.x);
            var randomPosZ = Random.Range(-m_SpawnAreaBounds.extents.z,
                m_SpawnAreaBounds.extents.z);

            var randomSpawnPos = spawnArea.transform.position +
                new Vector3(randomPosX, 0.45f, randomPosZ);

            Debug.Log($"pos x = {m_SpawnAreaBounds.extents.x}, pos z = {m_SpawnAreaBounds.extents.z})");
            Debug.Log($"Spawnpos: {randomSpawnPos}");
            return randomSpawnPos;
        }


        public void ResetBlock()
        {
            if (_connection.IsServer)
            {
                agent.transform.Find("shortBlock").SetPositionAndRotation(GetRandomSpawnPos(), _agentPrefab.transform.localRotation);
            }
        }


        public void ResetWall()
        {
            if (_connection.IsServer)
            {


                int config;
                config = Random.Range(0, 5);
                // config = 1;
                var localScale = agent.transform.Find("Wall").localScale;
                Debug.Log($"is server: {_connection.IsServer}. {localScale}");
                if (config == 0)
                {
                    localScale = new Vector3(
                        localScale.x,
                        0,
                        // m_ResetParams.GetWithDefault("no_wall_height", 0),
                        localScale.z);
                    agent.transform.Find("Wall").localScale = localScale;
                    // SetModel(m_NoWallBehaviorName, noWallBrain);
                }
                else if (config == 1)
                {
                    localScale = new Vector3(
                        localScale.x,
                        4,
                        // m_ResetParams.GetWithDefault("small_wall_height", 4),
                        localScale.z);
                    agent.transform.Find("Wall").localScale = localScale;
                    // SetModel(m_SmallWallBehaviorName, smallWallBrain);
                }
                else
                {
                    // var height = m_ResetParams.GetWithDefault("big_wall_height", 8);
                    localScale = new Vector3(
                        localScale.x,
                        8,
                        // height,
                        localScale.z);
                    agent.transform.Find("Wall").localScale = localScale;
                    
                }// SetModel(m_BigWallBehaviorName, bigWallBrain);
            }
        }

        // private void ResetObjects()
        // {
        //     // var spawnPoint = _map.FindSpawnPointForTreasure();
        //     _treasure.transform.SetPositionAndRotation(spawnPoint.center, _treasurePrefab.transform.rotation);
        // }



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
