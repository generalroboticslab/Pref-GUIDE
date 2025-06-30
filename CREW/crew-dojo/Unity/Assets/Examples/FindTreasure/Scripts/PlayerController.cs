using System;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements;
using Unity.Netcode;
using Dojo;
using Dojo.Netcode;

namespace Examples.FindTreasure
{
    public class PlayerController : NetworkBehaviour
    {
        private const string LOGSCOPE = "PlayerController";

        [SerializeField]
        private float _moveSpeed = 2f;

        [SerializeField, Tooltip("Rotation speed (in degrees)")]
        private float _rotateSpeed = 50f;

        [SerializeField]
        private UIDocument inGameUI;

        public Rigidbody _body;
        public Vector3 _offset;
        public Vector3 _angleOffset;

        private Camera _globalCamera;

        public event Action OnControllerReady;

        private DojoConnection _connection;
        private DojoTransport _transport;

        [SerializeField]
        private Camera _eyeCam;

        [SerializeField]
        private AccumuCamera _accCam;

        [SerializeField]
        private AccumuCamera _accCamSens;

        public AccumuCamera CamAcc => _accCam;
        public AccumuCamera CamAccSens => _accCamSens;

        [SerializeField]
        private Camera _egoCam;

        [SerializeField]
        private Camera _egoCamSens;

        [SerializeField]
        private InputActionAsset _playerActions;

        private InputActionMap _playerControl;

        [HideInInspector]
        public NetworkVariable<AIAction> humanAction = new NetworkVariable<AIAction>();

        private HumanInterface _humanInterface;

        private Vector3 last_pos;

        public NetworkVariable<bool> clear_cam_flag = new NetworkVariable<bool>(false);

        public NetworkVariable<int> cnt = new NetworkVariable<int>(0);

        private Unity.Netcode.Components.NetworkTransform networkTransform;


        private void Awake()
        {

            _body = GetComponentInChildren<Rigidbody>();
            networkTransform = GetComponent<Unity.Netcode.Components.NetworkTransform>();
            _offset = Vector3.zero;
            _angleOffset = Vector3.zero;

            _globalCamera = Camera.main;

            var uiRoot = inGameUI.rootVisualElement;
            inGameUI.rootVisualElement.style.display = DisplayStyle.None;

            _playerControl = _playerActions.actionMaps[0];
            _playerControl.Enable();

            _connection = FindObjectOfType<DojoConnection>();
            _humanInterface = FindObjectOfType<HumanInterface>();

            last_pos = _body.position;

            // clear_cam_flag.OnValueChanged += OnChangeCamFlag;




        }

        private void Update()
        {
            // Debug.Log($"{(_body.position - last_pos).magnitude}");
            // if ((_body.position - last_pos).magnitude > 0.6f)
            // {
            //     _accCam.ClearAccumulation();
            //     _accCamSens.ClearAccumulation();
            // }

            last_pos = _body.position;


            // _eyeCam.depth = -10;
            // var idx = UnityEngine.Random.Range(0, 3);
            // if (idx==0)
            // {
            //     ActionForward();
            // }
            // else if(idx==1)
            // {
            //     ActionBackward();
            // }
            // else if(idx==2)
            // {
            //     ActionRotateLeft();
            // }
            // else if(idx==3)
            // {
            //     ActionRotateRight();
            // }

            SwitchCameraIfNeeded();
            UpdateHumanAction();

            // Debug.Log($"{IsClient} {clear_cam_flag.Value}");


            if (clear_cam_flag.Value && IsClient)
            {
                Debug.Log($"{IsClient} clear cam");
                _accCam.ClearAccumulation();
                _accCamSens.ClearAccumulation();
                ChangeFlagServerRpc();
            }
        }

        [ServerRpc(RequireOwnership = false)]
        private void ChangeFlagServerRpc()
        {
            clear_cam_flag.Value = false;
        }

        private void FixedUpdate()
        {
            if (IsServer)
            {
                // Debug.Log($"offset: {_offset}");
                _body.velocity = _moveSpeed * _offset;
                _offset = Vector3.zero;

                _body.MoveRotation(_body.rotation * Quaternion.Euler(_angleOffset * Time.fixedDeltaTime));
                _angleOffset = Vector3.zero;


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
            if (true)//(IsClient)
            {
                // Debug.Log("acc cam enabled");
                _accCam.IsEnabled = true;
                _accCamSens.IsEnabled = true;
                // _egoCam.IsEnabled = true;
                // _egoCam.IsEnabled = true;

                _accCam.FollowGlobalCamera(_globalCamera);
                _accCamSens.FollowGlobalCamera(_globalCamera);
                // _egoCam.FollowGlobalCamera(_globalCamera);
                // _egoCamSens.FollowGlobalCamera(_globalCamera);
                // _egoCam.FollowAgent();
                // _egoCamSens.FollowAgent();
            }
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
                inGameUI.rootVisualElement.style.display = DisplayStyle.Flex;
            }
        }

        public override void OnLostOwnership()
        {
            if (IsClient)
            {
                Debug.Log($"{LOGSCOPE}: Lost Ownership");
            }
        }

        public void ActionForward()
        {

            // Debug.Log("ActionForward");
            if (IsServer)
            {
                _offset += transform.forward;
            }
            else
            {
                ActionForwardServerRpc();
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
            }
        }

        private void SwitchCameraIfNeeded()
        {
            if (!_connection.IsServer && !_humanInterface.IsWrittenFeedbackVisible)
            {
                if (_playerControl["Camera1"].IsPressed())
                {
                    _eyeCam.depth = -10;
                    _globalCamera.depth = 1;
                    _accCam.FullCamera.depth = -10;
                    _egoCam.depth = -10;
                }
                else if (_playerControl["Camera2"].IsPressed())
                {
                    _eyeCam.depth = 1;
                    _globalCamera.depth = -10;
                    _accCam.FullCamera.depth = -10;
                    _egoCam.depth = -10;

                }
                else if (_playerControl["Camera3"].IsPressed())
                {
                    _eyeCam.depth = -10;
                    _globalCamera.depth = -10;
                    _accCam.FullCamera.depth = 1;
                    _egoCam.depth = -10;
                    // Debug.Log($"{_accCam.FullCamera.depth} {_eyeCam.depth} {_globalCamera.depth}");
                }
                else if (_playerControl["Camera4"].IsPressed())
                {
                    _eyeCam.depth = -10;
                    _globalCamera.depth = -10;
                    _accCam.FullCamera.depth = -10;
                    _egoCam.depth = 1;
                }

                _accCamSens.FullCamera.depth = _accCam.FullCamera.depth;
                _egoCamSens.depth = _egoCam.depth;
            }
        }

        private void UpdateHumanAction()
        {
            if (_connection.IsClient)
            {
                if (_playerControl["Forward"].IsPressed())
                {
                    UpdateHumanActionServerRpc(AIAction.Forward);
                }
                else if (_playerControl["Backward"].IsPressed())
                {
                    UpdateHumanActionServerRpc(AIAction.Backward);
                }
                else if (_playerControl["RotateLeft"].IsPressed())
                {
                    UpdateHumanActionServerRpc(AIAction.TurnLeft);
                }
                else if (_playerControl["RotateRight"].IsPressed())
                {
                    UpdateHumanActionServerRpc(AIAction.TurnRight);
                }
                // else if (_playerControl["PickUp"].IsPressed())
                // {
                //     UpdateHumanActionServerRpc(AIAction.PickUp);
                // }
            }
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

        [ServerRpc(RequireOwnership = false)]
        private void UpdateHumanActionServerRpc(AIAction action)
        {
            humanAction.Value = action;
        }

        public void Teleport(Vector3 position)
        {
            if (IsServer)
            {
                // var rot = Quaternion.Euler(0, 90, 0);
                networkTransform.Teleport(position, transform.rotation, transform.localScale);
                //  networkTransform.Teleport(position , rot, transform.localScale);
            }
        }
    }
}