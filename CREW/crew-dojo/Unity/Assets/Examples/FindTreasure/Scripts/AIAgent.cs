using System;
using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using UnityEngine.InputSystem;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Dojo;
using System.Collections.Generic;
using System.Linq;
using Unity.Netcode;
using UnityEngine.AI;

namespace Examples.FindTreasure
{
    public class AIAgent : Agent
    {
        [Header("Configs")]

        [SerializeField] private int max_steps = 1000;
        [SerializeField] private bool _is_nav;
        [SerializeField] private bool _written_feedback = true;
        [SerializeField] private float _agentMoveSpeed = 4.0f;

        [SerializeField] private float _agentRotationSpeed = 60.0f;

        [Tooltip("Request decision every N seconds")]
        [SerializeField] private float _decisionRequestFrequency = 1.0f;
        [SerializeField] private bool _repeatActions = false;

        [HideInInspector] public int AgentID = -1;
        private PlayerController _playerController;

        private GameManager _gameManager;
        private AIAgentManager _agentManager;

        private Expert _expert;

        private bool _imitationLearning = false;

        private DojoConnection _connection;

        private float _feedback = 0;
        private int _fixedUpdateCount = 0;
        private bool _isActive = true;
        private int _trajectoryID = 0;

        private AIAction _lastAction;

        private int buffer = 0;
        private int steps = 0;
        private bool clear_cam_flag = false;

        private NavMeshAgent navmeshagent;

        private float running_feedback = 0;
        private int running_feedback_count = 0;

        private Vector2 _prevAction;

        // public CameraSensor cameraSensor;
        // private List<byte> _cameraData = new List<byte>();




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
                if (arg.Equals("-DecisionRequestFrequency") && idx < args.Length - 1 && float.TryParse(args[idx + 1], out var requestFreq))
                {
                    _decisionRequestFrequency = requestFreq;
                    ++idx;
                }

                InvokeRepeating(nameof(DecisionRequestLoop), 0.0f, _decisionRequestFrequency);
            }
#endif

            _connection = FindObjectOfType<DojoConnection>();
            _gameManager = FindObjectOfType<GameManager>();
            _agentManager = FindObjectOfType<AIAgentManager>();
            _expert = FindObjectOfType<Expert>();
            navmeshagent = GetComponentInChildren<NavMeshAgent>();


            if (_connection.IsServer)
            {
                _connection.SubscribeRemoteMessages((long)NetOpCode.Feedback, OnRemoteFeedback);
                _connection.SubscribeRemoteMessages((long)NetOpCode.ImitationLearning, OnImitationLearning);
                _playerController.SetMoveSpeed(_agentMoveSpeed);
                _playerController.SetRotationSpeed(_agentRotationSpeed);
            }

            // NavMeshAgent expert = GetComponent<NavMeshAgent>();
            // expert.destination = new Vector3(0, 0, 0);
            // expert.destination = _treasure.transform.position;
        }

        private void FixedUpdate()
        {

            // byte[] frameData = cameraSensor.GetCompressedObservation();
            // _cameraData.AddRange(frameData);

            running_feedback += _feedback;
            running_feedback_count += 1;
            // Debug.Log($"Feedback: {running_feedback}");
            var dis = (transform.Find("Body").position - _gameManager._treasure.transform.position).magnitude;
            // Debug.Log($"Distance to treasure: {dis}");
            if (dis < 2f)
            {
                Debug.Log("Treasure found!");
                AddReward(1);
                _isActive = false;
                EndEpisode();
                steps = 0;
                _trajectoryID += 1;
                _lastAction = 0;

                if (_is_nav)
                {
                    _expert.agent.isStopped = true;
                }
                else
                {
                    if (_written_feedback)
                    {
                        Debug.Log("--Sending Written!");
                        _connection.SendStateMessage((long)NetOpCode.ShowWrittenFeedback, "Show Written Feedback!");
                        Debug.Log("--Sending Written Done!");
                    }
                    else
                    {
                        _gameManager.OnReceiveWrittenFeedback(null);
                    }

                }

                if (_is_nav)
                {
                    _agentManager.ResetAgent(); // in non nav mode, this is done in GameManager.OnReceiveWrittenFeedback
                    _gameManager.ResetTreasure();
                    _expert.agent.enabled = true;
                    _expert.StartCoroutine(_expert.WaitAndSetDestination(new Vector2(0, 0)));
                }
            }

            // Debug.Log("fixed update");
            if (steps < max_steps)
            {
                steps += 1;
            }
            else
            {
                Debug.Log("Max Steps Reached");
                _isActive = false;
                EndEpisode();
                steps = 0;
                _trajectoryID += 1;
                _lastAction = 0;

                if (_is_nav)
                {
                    _expert.agent.isStopped = true;
                }
                else
                {
                    if (_written_feedback)
                    {
                        Debug.Log("--Sending Written!");
                        _connection.SendStateMessage((long)NetOpCode.ShowWrittenFeedback, "Show Written Feedback!");
                        Debug.Log("--Sending Written Done!");
                    }
                    else
                    {
                        _gameManager.OnReceiveWrittenFeedback(null);
                    }

                }


                if (_is_nav)
                {
                    _agentManager.ResetAgent(); // in non nav mode, this is done in GameManager.OnReceiveWrittenFeedback
                    _gameManager.ResetTreasure();
                    _expert.agent.enabled = true;
                    _expert.StartCoroutine(_expert.WaitAndSetDestination(new Vector2(0, 0)));
                }

            }

            // if (_is_nav)
            // {
            //     if (buffer == 5)
            //     {
            //         ExecuteAction((AIAction.Forward));
            //         buffer = 0;
            //     }

            //     buffer += 1;
            // }

            // else
            // {
            //     if (_repeatActions && _isActive && _connection.IsServer)
            //     {
            //         ExecuteAction(_lastAction);
            //     }
            // }\
            // if (_isActive && _connection.IsServer)
            //     {
            //         ExecuteAction(_prevAction);
            //     }

            // if (_connection.IsServer && _isActive)
            // {
            //     RequestDecision();
            // }
            // if (clear_cam_flag)
            // {
            //     _playerController.CamAcc.ClearAccumulation();
            //     _playerController.CamAccSens.ClearAccumulation();
            //     clear_cam_flag = false;
            // }
            _fixedUpdateCount += 1;
        }
        private void DecisionRequestLoop()
        {

            if (_isActive)
            {
                RequestDecision();
            }
        }

        // public void SubscribeController(PlayerController controller)
        // {
        //     var sensors = GetComponents<CameraSensorComponent>();

        //     foreach (var sensor in sensors)
        //     {
        //         sensor.Camera = _playerController.CamAccSens;
        //         sensor.enabled = true;
        //     }
        // }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Debug.Log("sent");
            // Debug.Log($"Feedback: {running_feedback / running_feedback_count}");
            // sensor.AddObservation(running_feedback / running_feedback_count); // use accumulated
            // sensor.AddObservation(running_feedback); // use accumulated
            sensor.AddObservation(_feedback);
            _feedback = 0;
            running_feedback = 0;
            running_feedback_count = 0;
            sensor.AddObservation(Time.realtimeSinceStartup);
            sensor.AddObservation(_trajectoryID);
            sensor.AddObservation(_imitationLearning);
            sensor.AddObservation((int)_playerController.humanAction.Value);
            sensor.AddObservation(_playerController.transform.position);
            sensor.AddObservation(_playerController.transform.rotation.eulerAngles);
            sensor.AddObservation(_gameManager._treasure.transform.position);
            // sensor.AddObservation(_cameraData);
            // _cameraData.Clear();

        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Debug.Log("action received");
            // ActionSegment<int> actSegment = actions.DiscreteActions;
            // var action = (AIAction)actSegment[0];


            // if (!_is_nav && _isActive)
            // {
            //     ExecuteAction(action);
            // }
            // actions = new Vector2(actionX, actionZ)
            var actionZ = actions.ContinuousActions[0];// * 10.0f;
            var actionX = actions.ContinuousActions[1];// * 10.0f;
            var action = new Vector2(actionX, actionZ);
            //Debug.Log($"Action received: {action}");

            // Debug.Log($"Action received: {action}");

            // transform.Find("VisDes").position = new Vector3(actionX, 0, actionZ);
            ExecuteAction(action);

        }

        private void ExecuteAction(Vector2 des)
        {
            if (!_connection.IsServer)
                return;


            //des = new Vector2(_gameManager._treasure.transform.position.x, _gameManager._treasure.transform.position.z);
            // var currnt_pos = new Vector2(_playerController.transform.position.x, _playerController.transform.position.z);
            // ActionNavigateToDes(curr + des);

            
            ActionNavigateToDes(des);

            // ActionSetVelocity(des);
            // _prevAction = des;
        }

        // private void ExecuteAction(AIAction action)
        // {
        //     // Debug.Log($"Action executed {action}");

        //     if (!_connection.IsServer)
        //         return;
        //     switch (action)
        //     {
        //         case AIAction.Forward:
        //             _playerController.ActionForward();
        //             break;
        //         case AIAction.Backward:
        //             _playerController.ActionBackward();
        //             break;
        //         case AIAction.TurnLeft:
        //             _playerController.ActionRotateLeft();
        //             break;
        //         case AIAction.TurnRight:
        //             _playerController.ActionRotateRight();
        //             break;
        // case AIAction.PickUp:
        //     var bounds = transform.Find("Body").GetComponent<Collider>().bounds;
        //     var colliders = Physics.OverlapBox(bounds.center, bounds.extents * 2);
        //     foreach (var collider in colliders)
        //     {
        //         if (collider.tag == "Treasure")
        //         {
        //             Debug.Log("Treasure found!");
        //             AddReward(1);
        //             _isActive = false;
        //             EndEpisode();
        //             _trajectoryID += 1;


        //             if (_is_nav)
        //             {
        //                 _expert.agent.isStopped = true;
        //             }
        //             else
        //             {
        //                 Debug.Log("--Sending Written!");
        //                 _connection.SendStateMessage((long)NetOpCode.ShowWrittenFeedback, "Show Written Feedback!");
        //                 Debug.Log("--Sending Written Done!");
        //             }


        //             if (_is_nav)
        //             {
        //                 _agentManager.ResetAgent(); // in non nav mode, this is done in GameManager.OnReceiveWrittenFeedback
        //                 _gameManager.ResetTreasure();
        //                 _expert.agent.enabled = true;
        //                 _expert.StartCoroutine(_expert.WaitAndSetDestination());
        //             }
        //             break;
        //         }
        //     }
        //     break;
        // }

        // var bounds = transform.Find("Body").GetComponent<Collider>().bounds;
        // var colliders = Physics.OverlapBox(bounds.center, bounds.extents * 2);
        // foreach (var collider in colliders)
        // {
        //     if (collider.tag == "Treasure")
        //     {
        //         Debug.Log("Treasure found!");
        //         AddReward(1);
        //         _isActive = false;
        //         EndEpisode();
        //         steps = 0;
        //         _trajectoryID += 1;
        //         _lastAction = 0;


        //         if (_is_nav)
        //         {
        //             _expert.agent.isStopped = true;
        //         }
        //         else
        //         {
        //             if (_written_feedback)
        //             {
        //                 Debug.Log("--Sending Written!");
        //                 _connection.SendStateMessage((long)NetOpCode.ShowWrittenFeedback, "Show Written Feedback!");
        //                 Debug.Log("--Sending Written Done!");
        //             }
        //             else{
        //                 _gameManager.OnReceiveWrittenFeedback(null);
        //             }

        //         }


        //         if (_is_nav)
        //         {
        //             _agentManager.ResetAgent(); // in non nav mode, this is done in GameManager.OnReceiveWrittenFeedback
        //             _gameManager.ResetTreasure();
        //             _expert.agent.enabled = true;
        //             _expert.StartCoroutine(_expert.WaitAndSetDestination());
        //         }
        //         break;
        //     }
        // }

        // _lastAction = action;
        // }



        private void OnRemoteFeedback(DojoMessage m)
        {
            var feedbackMessage = m.GetDecodedData<List<object>>();
            float feedback = Convert.ToSingle(feedbackMessage[0]);
            List<int> targets = (feedbackMessage[1] as IEnumerable<object>).Cast<object>().Cast<int>().ToList();
            if (targets.Contains(AgentID))
            {
                if (feedback != -9)
                _feedback = feedback;
                // _feedback = feedback;
                // if (_feedback == -9)
                // {
                //     _feedback = 0;
                // }
            }

        }

        private void OnImitationLearning(DojoMessage m)
        {
            if (!_connection.IsServer)
                return;
            var imitationLearningMessage = m.GetDecodedData<List<object>>();
            int target = (int)imitationLearningMessage[0];
            _imitationLearning = target == AgentID ? !_imitationLearning : false;
        }

        private void ActionNavigateToDes(Vector2 des)
        {
            if (_connection.IsServer)
            {
                navmeshagent.enabled = true;
                navmeshagent.SetDestination(new Vector3(des.x, 0, des.y));
                // _expert.StartCoroutine(_expert.WaitAndSetDestination(des));
            }
            else
            {
                ActionNavigateToDesServerRpc(des);
            }
        }

        [ServerRpc]
        private void ActionNavigateToDesServerRpc(Vector2 des)
        {
            ActionNavigateToDes(des);
        }

        private void ActionSetVelocity(Vector2 vel)
        {
            if (_connection.IsServer)
            {
                // vel = new Vector2(1, 1);
                vel = vel.normalized;

                _playerController._offset.x = vel.x * 0.6f;
                _playerController._offset.z = vel.y * 0.6f;
                // rotate the player along y direction to face the direction of the velocity (x, z).
                _playerController._body.transform.rotation = Quaternion.Euler(0, Mathf.Atan2(vel.x, vel.y) * Mathf.Rad2Deg, 0);

                // _playerController._body.AddForce(new Vector3(vel.x, 0, vel.y) * _agentMoveSpeed, ForceMode.VelocityChange);
            }
            else
            {
                ActionSetVelocityServerRpc(vel);
            }
        }

        [ServerRpc]
        private void ActionSetVelocityServerRpc(Vector2 vel)
        {
            ActionSetVelocity(vel);
        }


        public void StartRequestingDecisions()
        {

            if (!_connection.IsServer)
                return;
            _isActive = true;
        }

        public IEnumerator WaitAndStartRequestingDecisions()
        {
            yield return null; // waits one frame
            if (!_connection.IsServer)
                yield return null;
            _isActive = true;
        }

    }
}