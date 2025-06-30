using UnityEngine;
using Unity.Netcode;
using UnityEngine.InputSystem;

namespace Examples.Soccer
{
    public enum Team
    {
        Blue = 0,
        Purple = 1
    }

    public class PlayerController : NetworkBehaviour
    {
        public float rotateSpeed = 1.0f;
        public float forwardSpeed = 1.0f;

        [SerializeField]
        private InputActionAsset actionAsset;

        [SerializeField]
        private Camera playerCamera;

        private InputActionMap actionMap;

        private Rigidbody rigidBody;

        private Vector3 dirToGo = Vector3.zero;
        private Vector3 rotateDir = Vector3.zero;
        private float kickPower = 0.0f;

        private SoccerSettings soccerSettings;

        private Camera globalCamera;

        public Camera FirstPersonCamera => playerCamera;

        [HideInInspector] public Team team;

        private GameManager _gameManager;

        private void Awake()
        {
            rigidBody = GetComponent<Rigidbody>();
            soccerSettings = FindObjectOfType<SoccerSettings>();

            actionMap = actionAsset.actionMaps[0];
            actionMap.Enable();

            globalCamera = Camera.main;

            _gameManager = FindObjectOfType<GameManager>();
        }

        public override void OnNetworkSpawn()
        {
            if (IsOwner)
            {
                playerCamera.enabled = true;
                globalCamera.enabled = false;
            }
        }

        public override void OnNetworkDespawn()
        {
            playerCamera.enabled = false;

            if (IsOwner)
            {
                globalCamera.enabled = false;
            }
        }

        private void Update()
        {
            if (IsOwner)
            {
                if (actionMap["Forward"].IsPressed())
                {
                    HandleAction(AIAction.Forward);
                }
                if (actionMap["Backward"].IsPressed())
                {
                    HandleAction(AIAction.Backward);
                }
                if (actionMap["TurnLeft"].IsPressed())
                {
                    HandleAction(AIAction.TurnLeft);
                }
                if (actionMap["TurnRight"].IsPressed())
                {
                    HandleAction(AIAction.TurnRight);
                }
            }
        }

        private void FixedUpdate()
        {
            if (IsServer)
            {
                transform.Rotate(rotateDir, Time.deltaTime * 100f);
                rigidBody.AddForce(dirToGo * soccerSettings.agentRunSpeed, ForceMode.VelocityChange);

                dirToGo = Vector3.zero;
                rotateDir = Vector3.zero;
                kickPower = 0.0f;
            }
        }

        public void TurnOffAICamera()
        {
            if (IsServer)
            {
                playerCamera.enabled = false;
                globalCamera.enabled = true;
            }
        }

        public void HandleAction(AIAction action)
        {
            if (!_gameManager.IsGameRunning)
            {
                return;
            }

            dirToGo = Vector3.zero;
            rotateDir = Vector3.zero;

            switch (action)
            {
                case AIAction.Forward:
                    kickPower = 1f;
                    dirToGo = transform.forward * forwardSpeed;
                    break;
                case AIAction.Backward:
                    dirToGo = -transform.forward * forwardSpeed;
                    break;
                case AIAction.TurnLeft:
                    rotateDir = -transform.up * rotateSpeed;
                    break;
                case AIAction.TurnRight:
                    rotateDir = transform.up * rotateSpeed;
                    break;
            }

            if (IsOwner && IsClient)
            {
                HandleActionServerRpc(dirToGo, rotateDir, kickPower);
            }
        }

        [ServerRpc]
        private void HandleActionServerRpc(Vector3 dir, Vector3 rot, float power)
        {
            dirToGo = dir;
            rotateDir = rot;
            kickPower = power;
        }
    }
}