using System;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements;
using Unity.Netcode;
using Dojo;
using Dojo.Netcode;

using System.Collections;
using Unity.MLAgents;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgentsExamples;

using Random = UnityEngine.Random;



namespace Examples.WallJump
{
    public class PlayerController : NetworkBehaviour
    {
        private const string LOGSCOPE = "PlayerController";

        // Depending on this value, the wall will have different height

        [SerializeField]
        private float agentRunSpeed;

        [SerializeField]
        private float agentRotateSpeed;

        [SerializeField]
        private float agentJumpHeight;

        //when a goal is scored the ground will use this material for a few seconds.
        public Material goalScoredMaterial;
        //when fail, the ground will use this material for a few seconds.
        public Material failMaterial;

        [HideInInspector]
        public float agentJumpVelocity = 777;

        [HideInInspector]
        public float agentJumpVelocityMaxChange = 10;

        // [SerializeField]
        // private float _moveSpeed = 0.2f;

        // [SerializeField, Tooltip("Rotation speed (in degrees)")]
        // private float _rotateSpeed = 50f;

        [SerializeField]
        private UIDocument inGameUI;

        // public int m_Configuration;
        // Brain to use when no wall is present

        // public NNModel noWallBrain;
        // // Brain to use when a jumpable wall is present
        // public NNModel smallWallBrain;
        // // Brain to use when a wall requiring a block to jump over is present
        // public NNModel bigWallBrain;

        public GameObject ground;
        // public GameObject spawnArea;
        // Bounds m_SpawnAreaBounds;

        // [SerializeField]
        // private Camera playerCamera;

        [SerializeField]
        private InputActionAsset _playerActions;

        //  TODO: Add switch cameras to actions
        private InputActionMap _playerControl;

        [HideInInspector]
        public NetworkVariable<AIAction> humanAction = new NetworkVariable<AIAction>();

        private HumanInterface _humanInterface;

        private Camera _globalCamera;
        public event Action OnControllerReady;

        // public GameObject goal;

        [SerializeField]
        private GameObject shortBlock;
        // public GameObject wall;
        private Rigidbody m_ShortBlockRb;
        private Rigidbody m_AgentRb;
        private Material m_GroundMaterial;
        private Renderer m_GroundRenderer;

        // [SerializeField]
        // private WallJumpSettings m_WallJumpSettings;

        private DojoConnection _connection;
        private DojoTransport _transport;


        [SerializeField]
        private Camera _eyeCam;

        public float jumpingTime;
        public float jumpTime;
        // This is a downward force applied when falling to make jumps look
        // less floaty
        public float fallingForce;
        // Use to check the coliding objects
        public Collider[] hitGroundColliders = new Collider[3];
        private Vector3 m_JumpTargetPos;
        private Vector3 m_JumpStartingPos;

        // string m_NoWallBehaviorName = "SmallWallJump";
        // string m_SmallWallBehaviorName = "SmallWallJump";
        // string m_BigWallBehaviorName = "BigWallJump";

        EnvironmentParameters m_ResetParams;

        // private Rigidbody rigidBody;

        // private Vector3 _offset;
        // private Vector3 _angleOffset;
        private Vector3 dirToGo;
        private Vector3 rotateDir;

        private bool smallGrounded;
        private bool largeGrounded;

        private GameManager _gameManager;

        private bool doJump;



        private void Awake()
        {
            // m_WallJumpSettings = FindObjectOfType<WallJumpSettings>();

            // m_Configuration = Random.Range(0, 5);

            m_AgentRb = GetComponentInChildren<Rigidbody>();
            m_ShortBlockRb = shortBlock.GetComponent<Rigidbody>();
            // m_SpawnAreaBounds = spawnArea.GetComponent<Collider>().bounds;
            // m_GroundRenderer = ground.GetComponent<Renderer>();
            // m_GroundMaterial = m_GroundRenderer.material;

            // spawnArea.SetActive(false);

            // m_ResetParams = Academy.Instance.EnvironmentParameters;

            _globalCamera = Camera.main;


            var uiRoot = inGameUI.rootVisualElement;
            inGameUI.rootVisualElement.style.display = DisplayStyle.None;

            _playerControl = _playerActions.actionMaps[0];
            _playerControl.Enable();

            _connection = FindObjectOfType<DojoConnection>();
            _humanInterface = FindObjectOfType<HumanInterface>();


            // ResetBlock(m_ShortBlockRb);
            // transform.localPosition = new Vector3(7, 1, -9);
            // ResetAgent(m_AgentRb);

            // m_AgentRb.transform.Position = new Vector3(18 * (Random.value - 0.5f), 1, -9);
            // m_Configuration = Random.Range(0, 5);
            // m_AgentRb.velocity = default(Vector3);


            // Update model references if we're overriding
            // var modelOverrider = GetComponent<ModelOverrider>();
            // if (modelOverrider.HasOverrides)
            // {
            //     noWallBrain = modelOverrider.GetModelForBehaviorName(m_NoWallBehaviorName);
            //     m_NoWallBehaviorName = ModelOverrider.GetOverrideBehaviorName(m_NoWallBehaviorName);

            //     smallWallBrain = modelOverrider.GetModelForBehaviorName(m_SmallWallBehaviorName);
            //     m_SmallWallBehaviorName = ModelOverrider.GetOverrideBehaviorName(m_SmallWallBehaviorName);

            //     bigWallBrain = modelOverrider.GetModelForBehaviorName(m_BigWallBehaviorName);
            //     m_BigWallBehaviorName = ModelOverrider.GetOverrideBehaviorName(m_BigWallBehaviorName);
            // }


        }

        private void Update()
        {
            SwitchCameraIfNeeded();
            UpdateHumanAction();
            // ActionForward();
            // ActionRotateLeft();
            // ActionJump();
        }



        private void FixedUpdate()
        {
            // Debug.Log($"dirtogo{dirToGo}");
            // if (m_Configuration != -1)
            // {
            //     ConfigureAgent(m_Configuration);
            //     m_Configuration = -1;
            // }
            // var smallGrounded = DoGroundCheck(true);
            if (IsServer)
            {
                // m_AgentRb.MovePosition(transform.position + Time.deltaTime * agentRunSpeed * dirToGo);
                // dirToGo = Vector3.zero;

                // m_AgentRb.MoveRotation(m_AgentRb.rotation * Quaternion.Euler(rotateDir * Time.fixedDeltaTime));
                // rotateDir = Vector3.zero;
                var largeGrounded = DoGroundCheck(false);
                transform.Rotate(rotateDir, Time.fixedDeltaTime * 300f);
                m_AgentRb.AddForce(dirToGo * agentRunSpeed,
                ForceMode.VelocityChange);
                // dirToGo = Vector3.zero;
                rotateDir = Vector3.zero;

                if (jumpingTime > 0f)
                {
                    m_JumpTargetPos =
                        new Vector3(m_AgentRb.position.x,
                            m_JumpStartingPos.y + agentJumpHeight,
                            m_AgentRb.position.z) + dirToGo;
                    MoveTowards(m_JumpTargetPos, m_AgentRb, agentJumpVelocity,
                        agentJumpVelocityMaxChange);
                }

                if (!(jumpingTime > 0f) && !largeGrounded)
                {
                    m_AgentRb.AddForce(
                        Vector3.down * fallingForce, ForceMode.Acceleration);
                }
                jumpingTime -= Time.fixedDeltaTime;
                // m_AgentRb.velocity =  (largeGrounded ? 1f : 0.5f) * _moveSpeed * _offset;
                // _offset = Vector3.zero;

                // m_AgentRb.MoveRotation(m_AgentRb.rotation * Quaternion.Euler(_angleOffset * Time.fixedDeltaTime));
                // _angleOffset = Vector3.zero;
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
            // Debug.Log("Forward is called");
            var largeGrounded = DoGroundCheck(false);
            // dirToGo = Vector3.zero;
            // rotateDir = Vector3.zero;
            if (IsServer)
            {
                // _offset += transform.forward;
                dirToGo = (largeGrounded ? 1f : 0.5f) * 1f * transform.forward;
            }
            else
            {
                ActionForwardServerRpc();
            }
        }

        public void ActionBackward()
        {
            var largeGrounded = DoGroundCheck(false);
            // dirToGo = Vector3.zero;
            // rotateDir = Vector3.zero;
            if (IsServer)
            {
                dirToGo = (largeGrounded ? 1f : 0.5f) * -1f * transform.forward;
                // _offset -= transform.forward;
            }
            else
            {
                ActionBackwardServerRpc();
            }
        }

        public void ActionRotateLeft()
        {
            // dirToGo = Vector3.zero;
            // rotateDir = Vector3.zero;
            if (IsServer)
            {
                rotateDir = transform.up * -1f * agentRotateSpeed;
                // _angleOffset -= Vector3.up * _rotateSpeed;
            }
            else
            {
                ActionRotateLeftServerRpc();
            }
        }

        public void ActionRotateRight()
        {
            // dirToGo = Vector3.zero;
            // rotateDir = Vector3.zero;
            if (IsServer)
            {
                rotateDir = transform.up * 1f * agentRotateSpeed;
                // _angleOffset += Vector3.up * _rotateSpeed;
            }
            else
            {
                ActionRotateRightServerRpc();
            }
        }

        public void ActionJump()
        {
            smallGrounded = DoGroundCheck(true);
            if (IsServer)
            {
                if ((jumpingTime <= 0f) && smallGrounded)
                {
                    Jump();
                }
            }
            else
            {
                ActionJumpServerRpc();
            }
        }


        public void SetRotationSpeed(float speed)
        {
            if (IsServer)
            {
                agentRotateSpeed = speed;
                // _rotateSpeed = speed;
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
                agentRunSpeed = speed;
                // _moveSpeed = speed;
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
                }
                else if (_playerControl["Camera2"].IsPressed())
                {
                    _eyeCam.depth = 10;
                }
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
                else if (_playerControl["TurnLeft"].IsPressed())
                {
                    UpdateHumanActionServerRpc(AIAction.TurnLeft);
                }
                else if (_playerControl["TurnRight"].IsPressed())
                {
                    UpdateHumanActionServerRpc(AIAction.TurnRight);
                }
                else if (_playerControl["Jump"].IsPressed())
                {


                    UpdateHumanActionServerRpc(AIAction.Jump);
                }
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
        private void ActionJumpServerRpc()
        {

            ActionJump();
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

        public void Jump()
        {
            jumpingTime = 0.2f;
            m_JumpStartingPos = m_AgentRb.position;
        }

        public bool DoGroundCheck(bool smallCheck)
        {
            if (!smallCheck)
            {
                hitGroundColliders = new Collider[3];
                var o = gameObject;
                Physics.OverlapBoxNonAlloc(
                    o.transform.position + new Vector3(0, -0.05f, 0),
                    new Vector3(0.95f / 2f, 0.5f, 0.95f / 2f),
                    hitGroundColliders,
                    o.transform.rotation);
                var grounded = false;
                foreach (var col in hitGroundColliders)
                {
                    if (col != null && col.transform != transform &&
                        (col.CompareTag("walkableSurface") ||
                        col.CompareTag("block") ||
                        col.CompareTag("wall")))
                    {
                        grounded = true; //then we're grounded
                        break;
                    }
                }
                return grounded;
            }
            else
            {
                RaycastHit hit;
                Physics.Raycast(transform.position + new Vector3(0, -0.05f, 0), -Vector3.up, out hit,
                    1f);

                if (hit.collider != null &&
                    (hit.collider.CompareTag("walkableSurface") ||
                    hit.collider.CompareTag("block") ||
                    hit.collider.CompareTag("wall"))
                    && hit.normal.y > 0.95f)
                {
                    return true;
                }

                return false;
            }
        }

        private void MoveTowards(
            Vector3 targetPos, Rigidbody rb, float targetVel, float maxVel)
        {
            var moveToPos = targetPos - rb.worldCenterOfMass;
            var velocityTarget = Time.fixedDeltaTime * targetVel * moveToPos;
            if (float.IsNaN(velocityTarget.x) == false)
            {
                rb.velocity = Vector3.MoveTowards(
                    rb.velocity, velocityTarget, maxVel);
            }
        }

        // public override void CollectObservations(VectorSensor sensor)
        // {
        //     var agentPos = m_AgentRb.position - ground.transform.position;

        //     sensor.AddObservation(agentPos / 20f);
        //     sensor.AddObservation(DoGroundCheck(true) ? 1 : 0);
        // }



        // IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
        // {
        //     m_GroundRenderer.material = mat;
        //     yield return new WaitForSeconds(time); //wait for 2 sec
        //     m_GroundRenderer.material = m_GroundMaterial;
        // }
        // void OnTriggerStay(Collider col)
        // {
        //     if (col.gameObject.CompareTag("goal") && DoGroundCheck(true))
        //     {
        //         // SetReward(1f);
        //         // EndEpisode();
        //         // StartCoroutine(
        //         //     GoalScoredSwapGroundMaterial(m_WallJumpSettings.goalScoredMaterial, 2));
        //     }
        // }








    }

}
