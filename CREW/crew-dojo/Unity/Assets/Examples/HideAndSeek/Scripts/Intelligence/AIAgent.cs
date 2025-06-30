using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Dojo;
using System.Collections.Generic;
using System.Linq;

namespace Examples.HideAndSeek
{
    public class AIAgent : Agent
    {
        [Header("Configs")]
        [SerializeField] private float _agentMoveSpeed = 4.0f;

        [SerializeField] private float _agentRotationSpeed = 60.0f;

        [Tooltip("Request decision every N seconds")]
        [SerializeField] private float _decisionRequestFrequency = 1.0f;

        [SerializeField] private bool _repeatActions = true;

        [HideInInspector] public int AgentID = -1;
        private PlayerController _playerController;

        public bool IsPlayerAlive => _playerController != null;

        private AIAction _lastAction = AIAction.Forward;

        private GameManager _gameManager;
        private DojoConnection _connection;

        private float _feedbackReceived = 0;

        protected override void Awake()
        {
            base.Awake();
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
            }
#endif
            var sensors = GetComponents<CameraSensorComponent>();
            foreach (var sensor in sensors)
            {
                sensor.Camera = Camera.main;
            }
            _gameManager = GameManager.Instance;
            _connection = FindObjectOfType<DojoConnection>();
            _connection.SubscribeRemoteMessages((long)NetOpCode.Feedback, OnRemoteFeedback);
        }

        private void FixedUpdate()
        {
            if (_repeatActions && IsPlayerAlive)
            {
                ExecuteAction(_lastAction);
            }
        }

        private void DecisionRequestLoop()
        {
            bool gamePaused = _gameManager.GamePaused;
            if (IsPlayerAlive && !gamePaused)
            {
                RequestDecision();
            }
        }

        public void SubscribeController(PlayerController controller)
        {
            _playerController = controller;
            _playerController.AgentID = AgentID;
            _playerController.SetMoveSpeed(_agentMoveSpeed);
            _playerController.SetRotationSpeed(_agentRotationSpeed);

            var sensors = GetComponents<CameraSensorComponent>();
            foreach (var sensor in sensors)
            {
                if (sensor.SensorName.Contains("FirstPerson"))
                {
                    sensor.Camera = _playerController.CamEye;
                    sensor.enabled = _playerController.EnableFirstCamera;
                }
                else if (sensor.SensorName.Contains("Masked"))
                {
                    sensor.Camera = _playerController.CamMasked;
                    sensor.enabled = _playerController.EnableMaskedCamera;
                }
                else if (sensor.SensorName.Contains("Accumulative"))
                {
                    sensor.Camera = _playerController.CamAcc;
                    sensor.enabled = _playerController.EnableAccumuCamera;
                }
            }

            InvokeRepeating(nameof(DecisionRequestLoop), 0.0f, _decisionRequestFrequency);
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            sensor.AddObservation(_feedbackReceived);
            _feedbackReceived = 0;
            sensor.AddObservation(Time.realtimeSinceStartup);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            ActionSegment<int> actSegment = actions.DiscreteActions;
            var action = (AIAction)actSegment[0];
            ExecuteAction(action);
            _lastAction = action;
        }

        private void ExecuteAction(AIAction action)
        {
            bool gamePaused = _gameManager.GamePaused;
            if (!GameManager.Instance.GameRunning || !IsPlayerAlive || gamePaused)
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

                case AIAction.UpdateMap:
                    // TODO: how do we update?
                    break;
            }
        }

        private void OnRemoteFeedback(DojoMessage m)
        {
            var feedbackMessage = m.GetDecodedData<List<object>>();
            float feedback = Convert.ToSingle(feedbackMessage[0]);
            List<int> targets = (feedbackMessage[1] as IEnumerable<object>).Cast<object>().Cast<int>().ToList();
            if (targets.Contains(AgentID))
                _feedbackReceived += feedback;
        }
    }
}