using Dojo;
using Dojo.Netcode;
using Nakama;
using System.Collections.Generic;
using System.Linq;
using Unity.Netcode;
using UnityEngine;



namespace Examples.WallJump
{
    [DefaultExecutionOrder(-1)]
    public class GameManager : MonoBehaviour
    {

        private const string LOGSCOPE = "GameManager";
        // [SerializeField]
        // private GameObject prefabPlayer;

        // [SerializeField]
        // private GameObject prefabBox;

        [SerializeField]
        private GameObject environment;




        private readonly List<NetworkObject> Players = new();
        private NetworkObject Box;

        private DojoConnection _connection;
        private DojoTransport _transport;
        private bool IsClient => _connection.IsClient;

        private AIAgentManager _agentManager;

        // [SerializeField]
        // private GameObject _goalPrefab;
        [SerializeField]
        private GameObject _goal;

        // private bool _isFirstGame = true;

        #region Unity Lifecycle



        private void Awake()
        {


            Application.targetFrameRate = 60;
            QualitySettings.vSyncCount = 0;

            _connection = FindObjectOfType<DojoConnection>();
            _agentManager = GetComponentInChildren<AIAgentManager>();

            _connection.SubscribeRemoteMessages((long)NetOpCode.ReceiveWrittenFeedback, OnReceiveWrittenFeedback);
        }


        private void Start()
        {
            NetworkManager.Singleton.OnServerStarted += OnServerStarted;
        }


        #endregion Unity Lifecycle


        #region Callbacks

        private void OnServerStarted()
        {
            if (NetworkManager.Singleton.IsServer)
            {
                _transport = NetworkManager.Singleton.NetworkConfig.NetworkTransport as DojoTransport;

                // // load map on start
                // if (!_map.LoadMap(MapManager.DEFAULT_MAP))
                // {
                //     Debug.LogWarning($"{LOGSCOPE}: Invalid default map!");
                // }
                // else
                // {
                //     Debug.Log($"{LOGSCOPE}: Default map loaded!");
                // }

                _agentManager.SpawnAgent();

                // m_SpawnAreaBounds = spawnArea.GetComponent<Collider>().bounds;
                // m_Configuration = Random.Range(0, 5);

                // _goal = Instantiate(_goalPrefab);
                // _goal.GetComponent<NetworkObject>().Spawn();
                // Resetgoal(); 
            }
        }

        #endregion Callbacks

        public void ResetGame()
        {

        }

        // private void Resetgoal() {
        //     var spawnPoint = _map.FindSpawnPointForgoal();
        //     _goal.transform.SetPositionAndRotation(spawnPoint.center, _treasurePrefab.transform.rotation);
        // }



        private void OnReceiveWrittenFeedback(DojoMessage m)
        {
            if (!_connection.IsServer)
                return;

            
            _agentManager.ResetBlock();
            _agentManager.ResetWall();
            _agentManager.ResetAgent();



            // agent.StartRequestingDecisions();
            // _agentManager.ResetAgent();
            // ResetTreasure();
        }


        // private void OnPlayerJoinedOrRoleChanged(IUserPresence user)
        // {
        //     if (_connection.MatchClients.TryGetValue(user, out var role) && role == DojoNetworkRole.Player)
        //     {
        //         if (_transport.GetNetcodeIDByUser(user, out var userID))
        //         {
        //             SpawnPlayer(userID);
        //         }
        //     }
        // }

        // private void SpawnPlayer(ulong userID)
        // {
        //     if (NetworkManager.Singleton.IsServer)
        //     {

        //         if (Box == null)
        //         {
        //             var box = Instantiate(prefabBox, new Vector3(0.0f, 1.0f, 0.0f),
        //                 Quaternion.identity, environment.transform);
        //             var netObj = box.GetComponent<NetworkObject>();
        //             netObj.Spawn();

        //             // Box = netObj;
        //         }
        //         if (Players.Count<1)  
        //         {
        //             SpawnBluePlayer(userID);
        //         }

        //         // CheckEpisodeStart();
        //     }
        // }

        // private void SpawnBluePlayer(ulong userID)
        // {
        //     var posZ = Players.Count > 0 ? 1.0f : -1.0f;

        //     var player = Instantiate(prefabPlayer, new Vector3(2.0f, 1.0f, posZ),
        //                 Quaternion.LookRotation(new(1.0f, 0.0f, 0.0f), Vector3.up), environment.transform);

        //     var netObj = player.GetComponent<NetworkObject>();


        //     Debug.Log($"conected: {NetworkManager.Singleton.ConnectedClientsIds}");
        //     Debug.Log($"userid: {userID}");

        //     // netObj.Spawn();
        //     netObj.SpawnAsPlayerObject(userID);

        //     Players.Add(netObj);



        //     // Player = netObj;

        //     // bluePlayers.Add(netObj);
        // }

        // if (Box == null)
        // {
        //     var box = Instantiate(prefabBox, Vector3.zero,
        //         Quaternion.identity, environment.transform);
        //     var netObj = box.GetComponent<NetworkObject>();
        //     netObj.Spawn();
        //     Box = netObj;
        // }

        // public void CheckEpisodeStart()
        // {
        //     if (Player != null && Box != null)
        //     {
        //         Invoke(nameof(StartEpisode), 0f);
        //     }
        // }


    }
}