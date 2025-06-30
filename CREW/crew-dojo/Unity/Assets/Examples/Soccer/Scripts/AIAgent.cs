using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Dojo;
using System.Collections.Generic;
using System.Linq;

namespace Examples.Soccer
{
    public class AIAgent : Agent
    {
        [SerializeField]
        private float _rotateSpeed = 1f;

        [SerializeField]
        private float _forwardSpeed = 1.0f;

        [Tooltip("Request decision every N seconds")]
        [SerializeField] private float _decisionRequestFrequency = 1.0f;

        [SerializeField]
        private bool _repeatActions = true;

        [HideInInspector] public int AgentID = -1;
        private float _feedbackReceived = 0;

        private AIAction _lastAction;
        private PlayerController _controller;
        private DojoConnection _connection;

        public bool IsPlayerAlive => _controller != null;

        protected override void Awake()
        {
            base.Awake();
#if UNITY_STANDALONE // && !UNITY_EDITOR
            var args = Environment.GetCommandLineArgs();

            for (var idx = 0; idx < args.Length; ++idx)
            {
                var arg = args[idx];

                if (arg.Equals("-ForwardSpeed") && idx < args.Length - 1 && float.TryParse(args[idx + 1], out var moveSpeed))
                {
                    _forwardSpeed = moveSpeed;
                    ++idx;
                }

                if (arg.Equals("-RotationSpeed") && idx < args.Length - 1 && float.TryParse(args[idx + 1], out var rotSpeed))
                {
                    _rotateSpeed = rotSpeed;
                    ++idx;
                }

                if (arg.Equals("-DecisionRequestFrequency") && idx < args.Length - 1 && float.TryParse(args[idx + 1], out var requestFreq))
                {
                    _decisionRequestFrequency = requestFreq;
                    ++idx;
                }
            }
#endif
            _connection = FindObjectOfType<DojoConnection>();
            _connection.SubscribeRemoteMessages((long)NetOpCode.Feedback, OnRemoteFeedback);

            InvokeRepeating(nameof(DecisionRequestLoop), 0.0f, _decisionRequestFrequency);
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
            if (IsPlayerAlive)
            {
                RequestDecision();
            }
        }

        public void SubscribeController(PlayerController controller)
        {
            _controller = controller;
            _controller.rotateSpeed = _rotateSpeed;
            _controller.forwardSpeed = _forwardSpeed;

            var sensor = GetComponent<CameraSensorComponent>();
            sensor.Camera = _controller.FirstPersonCamera;

            _controller.TurnOffAICamera();
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
        }

        private void ExecuteAction(AIAction action)
        {
            _controller.HandleAction(action);
            _lastAction = action;
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