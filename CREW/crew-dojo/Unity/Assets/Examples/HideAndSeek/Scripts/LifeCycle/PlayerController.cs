using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements;
using Unity.Netcode;
using Nakama.TinyJson;
using Dojo;
using Dojo.Netcode;
using Dojo.Recording;

namespace Examples.HideAndSeek
{
    public class PlayerController : NetworkBehaviour
    {
        private const string LOGSCOPE = "PlayerController";

        [SerializeField]
        private float _moveSpeed = 2f;

        [SerializeField, Tooltip("Rotation speed (in degrees)")]
        private float _rotateSpeed = 50f;

        [SerializeField]
        private InputActionAsset _playerActions;

        [SerializeField]
        private MaskedCamera _maskedCamera;

        [SerializeField]
        private AccumuCamera _accumuCamera;

        [SerializeField]
        public Camera _egoCamera;

        [SerializeField]
        private UIDocument inGameUI;

        private Label _timeoutTextUI;
        private Label _hiderCountUI;
        private Label _seekerCountUI;
        private Label _identityUI;

        private InputActionMap _playerControl;

        private Rigidbody _body;
        private Vector3 _offset;
        private Vector3 _angleOffset;

        private Camera _globalCamera;

        [SerializeField]
        private Camera _firstPersonCamera;

        private IPlayer _selfPlayer;

        public Camera CamEye => _firstPersonCamera;
        public Camera CamMasked => _maskedCamera.EnvCamera;
        public Camera CamAcc => _accumuCamera.FullCamera;

        public event Action OnControllerReady;

        private DojoConnection _connection;
        private DojoRecord _record;
        private DojoTransport _transport;

        private bool _enableFirstCamera = true;
        public bool EnableFirstCamera => _enableFirstCamera;

        private bool _enableEgoCamera = true;
        public bool EnableEgoCamera => _enableEgoCamera;

        private bool _enableMaskedCamera = true;
        public bool EnableMaskedCamera => _enableMaskedCamera;
        private bool _enableAccumuCamera = true;
        public bool EnableAccumuCamera => _enableAccumuCamera;

        [HideInInspector] public int AgentID = -1;

        private GameManager _gameManager;

        private void Awake()
        {
            _playerControl = _playerActions.actionMaps[0];

            _body = GetComponentInChildren<Rigidbody>();
            _offset = Vector3.zero;
            _angleOffset = Vector3.zero;

            _globalCamera = Camera.main;

            // Camera[] camerasInChildren = GetComponentsInChildren<Camera>();

            // // Loop through all found Camera components
            // foreach (Camera camera in camerasInChildren)
            // {
            //     // Check if the GameObject of the Camera has the tag "Eye"
            //     if (camera.gameObject.CompareTag("Eye"))
            //     {
            //         _firstPersonCamera  = camera;
            //         Debug.Log("Eye camera found!");
            //         break; // Exit the loop once the camera is found
            //     }
            // }
            // _firstPersonCamera = GetComponentInChildren<Camera>();

            _selfPlayer = GetComponent<IPlayer>();

            var uiRoot = inGameUI.rootVisualElement;
            inGameUI.rootVisualElement.style.display = DisplayStyle.None;
            _timeoutTextUI = uiRoot.Q<Label>("Timeout");
            _hiderCountUI = uiRoot.Q<Label>("HiderCount");
            _seekerCountUI = uiRoot.Q<Label>("SeekerCount");
            _identityUI = uiRoot.Q<Label>("Identity");

            _connection = FindObjectOfType<DojoConnection>();
            _record = FindObjectOfType<DojoRecord>();

            _gameManager = GameManager.Instance;

#if UNITY_STANDALONE
            var args = Environment.GetCommandLineArgs();
            for (var idx = 0; idx < args.Length; ++idx)
            {
                var arg = args[idx];
                if (arg.Equals("-DisableFirstCamera"))
                {
                    _enableFirstCamera = false;
                }
                else if (arg.Equals("-DisableMaskedCamera"))
                {
                    _enableMaskedCamera = false;
                }
                else if (arg.Equals("-DisableAccumuCamera"))
                {
                    _enableAccumuCamera = false;
                }
            }
#endif
        }

        private void Update()
        {
            if (IsOwner)
            {
                HandleHumanInput();
                _timeoutTextUI.text = TimeSpan.FromSeconds(GameManager.Instance.GetMatchTimeout()).ToString(@"mm\:ss");
                _hiderCountUI.text = $"Hiders: {GameManager.Instance.HiderCount}";
                _seekerCountUI.text = $"Seekers: {GameManager.Instance.SeekerCount}";
            }
        }

        private void FixedUpdate()
        {
            if (IsServer)
            {
                _body.MovePosition(transform.position + Time.deltaTime * _moveSpeed * _offset);
                _offset = Vector3.zero;

                _body.MoveRotation(_body.rotation * Quaternion.Euler(_angleOffset * Time.fixedDeltaTime));
                _angleOffset = Vector3.zero;
            }
        }

        public override void OnDestroy()
        {
            try
            {
                _globalCamera.enabled = true;
            }
            catch { }
        }

        private void OnCollisionEnter(Collision other)
        {
            var player = other.gameObject.GetComponent<IPlayer>();
            var playerController = other.gameObject.GetComponent<PlayerController>();
            if (IsServer && player != null)
            {
                if (_transport.GetUserByNetcodeID(OwnerClientId, out var user1) &&
                    _connection.MatchClients.TryGetValue(user1, out var role1) &&
                    _transport.GetUserByNetcodeID(playerController.OwnerClientId, out var user2) &&
                    _connection.MatchClients.TryGetValue(user2, out var role2))
                {
                    var eventData = new List<string>() { user1.UserId, role1.ToString(), user2.UserId, role2.ToString() };
                    _record.DispatchEvent(RecordEvent.PlayerCollision, JsonWriter.ToJson(eventData));
                }

                if (_selfPlayer.IsHider && !player.IsHider)
                {
                    Debug.Log("Hider collide");

                    if (AgentID >= 0)
                    {
                        GameManager.Instance.HiderHasDied(AgentID);
                    }
                    else if (GameManager.Instance.HostType == GameHostType.WaitingRoom)
                    {
                        GameManager.Instance.HiderHasDied(OwnerClientId);
                    }
                    else
                    {
                        HiderCollidedClientRpc();
                    }
                }
                else if (!_selfPlayer.IsHider && player.IsHider)
                {
                    Debug.Log("Seeker collide");

                    var controller = player.GetComponent<PlayerController>();
                    GameManager.Instance.SeekerHasCaught(AgentID, controller.AgentID);
                }
            }
        }

        public override void OnNetworkSpawn()
        {
            if (IsOwner && IsClient)
            {
                OnGainedOwnership();
            }
            OnControllerReady?.Invoke();

            _transport = NetworkManager.Singleton.NetworkConfig.NetworkTransport as DojoTransport;
        }

        public override void OnNetworkDespawn()
        {
            if (IsOwner && IsClient)
            {
                OnLostOwnership();
            }
        }

        public override void OnGainedOwnership()
        {
            if (IsClient)
            {
                Debug.Log($"{LOGSCOPE}: Gained Ownership");
                _globalCamera.enabled = false;
                _firstPersonCamera.enabled = _enableFirstCamera;
                _egoCamera.enabled = false;
                _playerControl.Enable();
                _accumuCamera.IsEnabled = _enableAccumuCamera;
                _accumuCamera.FollowGlobalCamera(_globalCamera);
                _maskedCamera.IsEnabled = _enableMaskedCamera;
                _maskedCamera.FollowGlobalCamera(_globalCamera);
                inGameUI.rootVisualElement.style.display = DisplayStyle.Flex;
                _identityUI.text = "You're " + (GameManager.Instance.IsHider ? "Hider" : "Seeker");
                SwitchCamera(0);
                _record.DispatchEvent(RecordEvent.PlayerStateChange, $"Spawned {_rotateSpeed} {_moveSpeed} {DojoRecordEncode.Encode(transform)}");
            }
        }

        public override void OnLostOwnership()
        {
            if (IsClient)
            {
                Debug.Log($"{LOGSCOPE}: Lost Ownership");
                _globalCamera.enabled = true;
                _firstPersonCamera.enabled = false;
                _egoCamera.enabled = false;
                _playerControl.Disable();
            }
        }

        private void SwitchCamera(int cameraIdx)
        {
            if (cameraIdx == 1 && !_enableFirstCamera)
            {
                return;
            }
            else if (cameraIdx == 2 && !_enableMaskedCamera)
            {
                return;
            }
            else if (cameraIdx == 3 && !_enableAccumuCamera)
            {
                return;
            }

            TurnOffCamera(_firstPersonCamera);
            TurnOffCamera(_maskedCamera.EnvCamera);
            TurnOffCamera(_accumuCamera.FullCamera);
            TurnOffCamera(_egoCamera);

            switch (cameraIdx)
            {
                case 2:
                    {
                        TurnOnCamera(_maskedCamera.EnvCamera);
                        break;
                    }

                case 3:
                    {
                        TurnOnCamera(_accumuCamera.FullCamera);
                        break;
                    }
                case 4:
                    {
                        _egoCamera.enabled = true;
                        TurnOnCamera(_egoCamera);
                        break;
                    }

                case 1:
                default:
                    {
                        TurnOnCamera(_firstPersonCamera);
                        break;
                    }
            }
        }

        private void TurnOnCamera(Camera cam)
        {
            cam.depth = 1.0f;
            //cam.clearFlags = CameraClearFlags.Skybox;
            //cam.cullingMask = LayerMask.GetMask(new[] { "Everything", "Default", "TransparentFX", "Ignore Raycast", "PostProcessing", "Water", "UI", "Players" });
        }

        private void TurnOffCamera(Camera cam)
        {
            cam.depth = -100.0f;
            //cam.clearFlags = CameraClearFlags.Color;
            //cam.cullingMask = LayerMask.GetMask(new[] { "Nothing" });
        }

        private void HandleHumanInput()
        {
            bool gamePaused = _gameManager.GamePaused;
            if (_playerControl["Forward"].IsPressed() && !gamePaused)
            {
                ActionForward();
            }
            if (_playerControl["Backward"].IsPressed() && !gamePaused)
            {
                ActionBackward();
            }
            if (_playerControl["RotateLeft"].IsPressed() && !gamePaused)
            {
                ActionRotateLeft();
            }
            if (_playerControl["RotateRight"].IsPressed() && !gamePaused)
            {
                ActionRotateRight();
            }
            if (_playerControl["Camera1"].WasPerformedThisFrame())
            {
                SwitchCamera(1);
            }
            if (_playerControl["Camera2"].WasPerformedThisFrame())
            {
                SwitchCamera(2);
            }
            if (_playerControl["Camera3"].WasPerformedThisFrame())
            {
                SwitchCamera(3);
            }
            if (_playerControl["Camera4"].WasPerformedThisFrame())
            {
                SwitchCamera(4);
            }
        }

        public void ActionForward()
        {
            if (IsServer)
            {
                _offset += transform.forward;
            }
            else
            {
                ActionForwardServerRpc();
                RecordAction("Forward");
            }
        }

        public void ActionBackward()
        {
            if (IsServer)
            {
                _offset -= transform.forward;
            }
            else
            {
                ActionBackwardServerRpc();
                RecordAction("Backward");
            }
        }

        public void ActionRotateLeft()
        {
            if (IsServer)
            {
                _angleOffset -= Vector3.up * _rotateSpeed;
            }
            else
            {
                ActionRotateLeftServerRpc();
                RecordAction("RotateLeft");
            }
        }

        public void ActionRotateRight()
        {
            if (IsServer)
            {
                _angleOffset += Vector3.up * _rotateSpeed;
            }
            else
            {
                ActionRotateRightServerRpc();
                RecordAction("RotateRight");
            }
        }

        public void SetRotationSpeed(float speed)
        {
            if (IsServer)
            {
                _rotateSpeed = speed;
            }
            else
            {
                SetRotationSpeedServerRpc(speed);
                _record.DispatchEvent(RecordEvent.PlayerStateChange, $"RotationSpeed {speed}");
            }
        }

        public void SetMoveSpeed(float speed)
        {
            if (IsServer)
            {
                _moveSpeed = speed;
            }
            else
            {
                SetMoveSpeedServerRpc(speed);
                _record.DispatchEvent(RecordEvent.PlayerStateChange, $"MoveSpeed {speed}");
            }
        }

        private void RecordAction(string action)
        {
            _record.DispatchEvent(RecordEvent.PlayerAction, $"{action} {DojoRecordEncode.Encode(transform)}");
        }

        [ServerRpc]
        private void ActionForwardServerRpc()
        {
            ActionForward();
        }

        [ServerRpc]
        private void ActionBackwardServerRpc()
        {
            ActionBackward();
        }

        [ServerRpc]
        private void ActionRotateLeftServerRpc()
        {
            ActionRotateLeft();
        }

        [ServerRpc]
        private void ActionRotateRightServerRpc()
        {
            ActionRotateRight();
        }

        [ServerRpc]
        private void SetRotationSpeedServerRpc(float speed)
        {
            SetRotationSpeed(speed);
        }

        [ServerRpc]
        private void SetMoveSpeedServerRpc(float speed)
        {
            SetMoveSpeed(speed);
        }

        [ClientRpc]
        private void HiderCollidedClientRpc()
        {
            if (IsOwner && GameManager.Instance.HostType == GameHostType.JoinLeave)
            {
                GameManager.Instance.HiderHasDied();
            }
        }
    }
}