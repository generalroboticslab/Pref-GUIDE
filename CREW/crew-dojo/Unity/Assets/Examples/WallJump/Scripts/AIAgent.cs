using System;
using UnityEngine;
using Unity.MLAgents;
using UnityEngine.InputSystem;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Dojo;
using System.Collections.Generic;
using System.Linq;
using Unity.Netcode;
using Random = UnityEngine.Random;

namespace Examples.WallJump
{

    public class AIAgent : Agent
    {
        [Header("Configs")]
        [SerializeField] private float _agentMoveSpeed = 4.0f;

        [SerializeField] private float _agentRotationSpeed = 60.0f;

        [SerializeField] private bool _repeatActions = true;

        [HideInInspector] public int AgentID = -1;
        private PlayerController _playerController;

        private bool _imitationLearning = false;

        private DojoConnection _connection;

        private float _feedback = 0;
        private int _fixedUpdateCount = 0;
        private bool _isActive = true;
        private int _trajectoryID = 0;
        
        private AIAgentManager _agentManager;


        protected override void Awake()
        {
            base.Awake();

            _playerController = GetComponentInChildren<PlayerController>();

#if UNITY_STANDALONE // && !UNITY_EDITOR
            var args = Environment.GetCommandLineArgs();

            for (var idx = 0; idx < args.Length; ++idx)
            {
                var arg = args[idx];

                if (arg.Equals("-MoveSpeed") && idx < args.Length - 1 && float.TryParse(args[idx + 1], out var moveSpeed))
                {
                    _agentMoveSpeed = moveSpeed;
                    ++idx;
                }

                if (arg.Equals("-RotationSpeed") && idx < args.Length - 1 && float.TryParse(args[idx + 1], out var rotSpeed))
                {
                    _agentRotationSpeed = rotSpeed;
                    ++idx;
                }
            }
#endif

            _connection = FindObjectOfType<DojoConnection>();
            _agentManager = FindObjectOfType<AIAgentManager>();

            if (_connection.IsServer)
            {
                _connection.SubscribeRemoteMessages((long)NetOpCode.Feedback, OnRemoteFeedback);
                _connection.SubscribeRemoteMessages((long)NetOpCode.ImitationLearning, OnImitationLearning);
                _playerController.SetMoveSpeed(_agentMoveSpeed);
                _playerController.SetRotationSpeed(_agentRotationSpeed);
            }
        }

        private void FixedUpdate()
        {
            if (_connection.IsServer && _isActive)
            {
                RequestDecision();
            }
            _fixedUpdateCount += 1;
            Debug.Log($"is imit: {_imitationLearning}");


            if (_connection.IsServer && _isActive)
            {
                var bounds = transform.Find("GreenPlayer").GetComponent<Collider>().bounds;
                var colliders = Physics.OverlapBox(bounds.center, bounds.extents * 1.1f);
                foreach (var collider in colliders)
                {
                    if (collider.tag == "goal")
                    {
                        Debug.Log("goal detected");
                        AddReward(1);
                        _isActive = false;
                        EndEpisode();
                        _trajectoryID += 1;
                        _agentManager.ResetBlock();
                        _agentManager.ResetWall();
                        _agentManager.ResetAgent();

                        // _connection.SendStateMessage((long)NetOpCode.ShowWrittenFeedback, "Show Written Feedback!");
                        break;
                    }
                }
            }
            // break;

        }

        // void OnTriggerStay(Collider col)
        // {
        //     if (col.gameObject.CompareTag("goal") && _playerController.DoGroundCheck(true))
        //     {
        //         AddReward(1f);
        //         EndEpisode();
        //         // StartCoroutine(
        //         //     GoalScoredSwapGroundMaterial(m_WallJumpSettings.goalScoredMaterial, 2));
        //     }
        // }
        // public override void OnEpisodeBegin()
        // {
        //     _playerController.ResetBlock(_playerController.m_ShortBlockRb);
        //     _playerController.m_AgentRb.transform.localPosition = new Vector3(
        //         18 * (Random.value - 0.5f), 1, -9);
        //     _playerController.m_Configuration = Random.Range(0, 5);
        //     _playerController.m_AgentRb.velocity = default(Vector3);
        // }


        public override void CollectObservations(VectorSensor sensor)
        {
            sensor.AddObservation(_feedback);
            sensor.AddObservation(_imitationLearning);
            sensor.AddObservation((int)_playerController.humanAction.Value);
            sensor.AddObservation(_trajectoryID);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            ActionSegment<int> actSegment = actions.DiscreteActions;
            var action = (AIAction)actSegment[0];
            ExecuteAction(action);
        }

        private void ExecuteAction(AIAction action)
        {
            if (!_connection.IsServer)
                return;
            switch (action)
            {
                case AIAction.Forward:
                    _playerController.ActionForward();
                    break;
                case AIAction.Backward:
                    _playerController.ActionBackward();
                    break;
                case AIAction.TurnLeft:
                    _playerController.ActionRotateLeft();
                    break;
                case AIAction.TurnRight:
                    _playerController.ActionRotateRight();
                    break;
                case AIAction.Jump:
                    _playerController.ActionJump();
                    break;
            }
        }

        private void OnRemoteFeedback(DojoMessage m)
        {
            var feedbackMessage = m.GetDecodedData<List<object>>();
            float feedback = Convert.ToSingle(feedbackMessage[0]);
            List<int> targets = (feedbackMessage[1] as IEnumerable<object>).Cast<object>().Cast<int>().ToList();
            if (targets.Contains(AgentID))
            {
                _feedback = feedback;
            }
        }

        private void OnImitationLearning(DojoMessage m)
        {
            if (!_connection.IsServer)
                return;
            var imitationLearningMessage = m.GetDecodedData<List<object>>();
            int target = (int)imitationLearningMessage[0];
            _imitationLearning = target == AgentID ? !_imitationLearning : false;
            // Debug.Log($"{target}, {AgentID}, {_imitationLearning}");
            // Debug.Log($"is imit: {_imitationLearning}");

        }

        public void StartRequestingDecisions()
        {
            if (!_connection.IsServer)
                return;
            _isActive = true;
        }


        // public override void Heuristic()
        // {
        //     // var discreteActionsOut = actionsOut.DiscreteActions;

        //     ActionForward();


        //     ActionBackward();

        //     ActionRotateLeft();

        //     ActionJump();

        //     // discreteActionsOut[3] = Input.GetKey(KeyCode.Space) ? 1 : 0;
        // }

    }




}
