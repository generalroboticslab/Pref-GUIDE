using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Netcode;
using Dojo;
using Dojo.Netcode;
using Nakama.TinyJson;
using Unity.MLAgents;
using UnityEngine.AI;
using Unity.Netcode.Components;
// using System.Text.Json;

using System.IO;

namespace Examples.FindTreasure
{
    [Serializable]
    public class StringsList
    {
        public List<string> strings;
    }
    [DefaultExecutionOrder(-1)]
    public class GameManager : MonoBehaviour
    {
        private const string LOGSCOPE = "GameManager";

        [SerializeField]
        private MapManager _map;

        private DojoConnection _connection;
        private DojoTransport _transport;

        private bool IsClient => _connection.IsClient;

        private AIAgentManager _agentManager;

        [SerializeField]
        private GameObject _treasurePrefab;

        public GameObject _treasure;

        private bool _isFirstGame = true;

        private PlayerController _playerController;
        private NetworkTransform _networkTransform;

        private System.Random rnd = new System.Random();

        // #if UNITY_EDITOR
        //     static string jsonString = File.ReadAllText("Assets/Examples/FindTreasure/Scripts/new_maps.json");
        // #else
        //     static string jsonString = File.ReadAllText("../crew-dojo/Unity/Assets/Examples/FindTreasure/Scripts/new_maps.json");
        // #endif
        //     StringsList maps = JsonUtility.FromJson<StringsList>(jsonString);

        #region Unity Lifecycle

        private void Awake()
        {
            Application.targetFrameRate = 60;
            QualitySettings.vSyncCount = 0;

            _connection = FindObjectOfType<DojoConnection>();
            _agentManager = GetComponentInChildren<AIAgentManager>();
            _networkTransform = _treasurePrefab.GetComponent<NetworkTransform>();


            _connection.SubscribeRemoteMessages((long)NetOpCode.ReceiveWrittenFeedback, OnReceiveWrittenFeedback);

        }

        private void Start()
        {
            // Debug.Log(maps.strings[0]);

            NetworkManager.Singleton.OnServerStarted += OnServerStarted;
        }

        #endregion Unity Lifecycle


        #region Callbacks

        private void OnServerStarted()
        {
            if (NetworkManager.Singleton.IsServer)
            {
                _transport = NetworkManager.Singleton.NetworkConfig.NetworkTransport as DojoTransport;

                // load map on start
                if (!_map.LoadMap(MapManager.DEFAULT_MAP))
                // if (!_map.LoadMap(maps.strings[rnd.Next(0, maps.strings.Count)]))
                {
                    Debug.LogWarning($"{LOGSCOPE}: Invalid default map!");
                }
                else
                {
                    Debug.Log($"{LOGSCOPE}: Default map loaded!");
                }

                _treasure = Instantiate(_treasurePrefab);
                _treasure.GetComponent<NetworkObject>().Spawn();
                ResetTreasure();
                _agentManager.SpawnAgent();
                _agentManager.ResetAgent();

            }
        }

        #endregion Callbacks

        public void ResetGame()
        {

        }

        public void ResetTreasure()
        {

            // _map.LoadMap(maps.strings[rnd.Next(0, maps.strings.Count)]);
            var spawnPoint = _map.FindSpawnPointForTreasure();

            _treasure.transform.SetPositionAndRotation(spawnPoint.center, _treasurePrefab.transform.rotation);
            // var pos = new Vector3(0, 0, 0);
            // _treasure.transform.SetPositionAndRotation(pos, _treasurePrefab.transform.rotation);



            // if (!IsClient)
            // {
            //     // _networkTransform.Teleport(spawnPoint.center, _treasurePrefab.transform.localRotation, _treasurePrefab.transform.localScale);
            //     _treasure.transform.SetPositionAndRotation(spawnPoint.center, _treasurePrefab.transform.rotation);
            // }
            // _networkTransform.Teleport(spawnPoint.center, _treasurePrefab.transform.localRotation, _treasurePrefab.transform.localScale);
        }

        public void OnReceiveWrittenFeedback(DojoMessage m)
        {

            // _agentManager.ClearScreen();
            if (!_connection.IsServer)
                return;

            bool successful = false;
            while (!successful)
            {
                try
                {
                    // Attempt your operation here
                    Debug.Log("resetting");
                    ResetTreasure();
                    _agentManager.ResetAgent();
                    successful = true; // Set to true if operation succeeds
                }
                catch (Exception ex)
                {
                    Debug.Log(ex.Message);
                }
            }

        }

    }
}