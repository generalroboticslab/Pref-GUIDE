using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Dojo;
using System.Collections.Generic;
using System.Linq;

namespace Examples.Tetris
{
    public class AIAgent : Agent
    {
        private GameBoard _board;
        private DojoConnection _connection;
        private int AgentID = 0;
        // private float _feedbackReceived = 0;

        private bool _optionSpecialMode = false;
        private bool _isDone = false;
        private int trajID = 0;

        private float _feedback = 0;


        private float running_feedback = 0;
        private int running_feedback_count = 0;

        protected override void Awake()
        {
            base.Awake();
#if UNITY_STANDALONE // && !UNITY_EDITOR
            var args = Environment.GetCommandLineArgs();

            for (var idx = 0; idx < args.Length; ++idx)
            {
                var arg = args[idx];

                // "default" or "special"
                if (arg.Equals("-TetrisMode") && idx < args.Length - 1 && args[idx + 1].Contains("special"))
                {
                    _optionSpecialMode = true;
                    ++idx;
                }
            }
#endif

            _board = FindObjectOfType<GameBoard>();
            _board.OnScoreUpdate += OnGameScoreUpdate;
            _board.OnGameOver += OnGameOverEvent;

            _connection = FindObjectOfType<DojoConnection>();
            _connection.SubscribeRemoteMessages((long)NetOpCode.Feedback, OnRemoteFeedback);
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // sensor.AddObservation(running_feedback / running_feedback_count); // use accumulated
            sensor.AddObservation(_feedback);
            _feedback = 0;
            running_feedback = 0;
            running_feedback_count = 0;
            sensor.AddObservation(Time.realtimeSinceStartup);
            sensor.AddObservation(trajID);
            sensor.AddObservation(_board.EncodeStateForUnityAgents());
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            ActionSegment<int> actSegment = actions.DiscreteActions;
            var action = (AIAction)actSegment[0];
            switch (action)
            {
                case AIAction.MOVE_LEFT:
                    _board.HandleClientControl(NetCommand.MoveLeft);
                    break;
                case AIAction.MOVE_RIGHT:
                    _board.HandleClientControl(NetCommand.MoveRight);
                    break;
                case AIAction.ROTATE:
                    _board.HandleClientControl(NetCommand.Rotate);
                    break;
                case AIAction.FLIP:
                    _board.HandleClientControl(NetCommand.Flip);
                    break;
                case AIAction.DROP:
                    _board.HandleClientControl(NetCommand.Drop);
                    break;
                default:
                    break;
            }
            if (_isDone)
            {
                EndEpisode();
                _isDone = false;
                trajID++;
            }
        }

        private void FixedUpdate()
        {
            running_feedback += _feedback;
            running_feedback_count += 1;
        }

        private void OnGameScoreUpdate(int newScore, int increase)
        {
            AddReward(increase);
        }

        private void OnGameOverEvent()
        {
            _isDone = true;
        }

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
    }
}