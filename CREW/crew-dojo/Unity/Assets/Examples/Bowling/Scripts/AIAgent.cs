using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System;
using Dojo;
using System.Collections.Generic;
using System.Linq;

namespace Examples.Bowling
{
    public class AIAgent : Agent
    {
        private GameBoard _board;
        private DojoConnection _connection;
        private int AgentID = 0;
        // private float _feedbackReceived = 0;

        private bool _isDone = false;
        private int trajID = 0;

        [Tooltip("Request decision every N seconds")]
        [SerializeField] private float _decisionRequestFrequency = 1f;
        [SerializeField] private bool _repeatActions = true;
        // private AIAction _lastAction = AIAction.NO_MOVE;
        private Vector3 _lastAction = Vector3.zero;
        private int decisionfreqcounter = 0;


        private float _feedback = 0;



        private float running_feedback = 0;
        private int running_feedback_count = 0;

        protected override void Awake()
        {
            base.Awake();
            _board = FindObjectOfType<GameBoard>();
            _board.OnEpisodeEnded += OnGameEnded;
            _board.OnFrameEnded += OnFrameEnded;

            _connection = FindObjectOfType<DojoConnection>();

            if (_connection.IsServer)
            {
                _connection.SubscribeRemoteMessages((long)NetOpCode.Feedback, OnRemoteFeedback);
            }

#if UNITY_STANDALONE // && !UNITY_EDITOR
            var args = Environment.GetCommandLineArgs();

            for (var idx = 0; idx < args.Length; ++idx)
            { 
                var arg = args[idx];                                                                                                                                                                                                                                                                                                                                              

                if (arg.Equals("-DecisionRequestFrequency") && idx < args.Length - 1 && float.TryParse(args[idx + 1], out var requestFreq))
                {
                    _decisionRequestFrequency = requestFreq;
                    ++idx;
                }
            }
#endif
            // InvokeRepeating(nameof(DecisionRequestLoop), 0.0f, _decisionRequestFrequency);
        }

        private void FixedUpdate()
        {
            running_feedback += _feedback;
            running_feedback_count += 1;
            if (_repeatActions)
            {
                ExecuteAction(_lastAction);
            }
            
            if (_board.newReward)
            {
                AddReward(_board.immediate_score);
                _board.newReward = false;
            }
            // Log decirion freq
            // Debug.Log("Decision freq: " + decisionfreqcounter + " " + _decisionRequestFrequency/0.05);
            // if (decisionfreqcounter > (_decisionRequestFrequency/0.02) || _board._stage == GameStage.Move)
            // {
            //     decisionfreqcounter = 0;
            //     RequestDecision();
            // }
            // decisionfreqcounter += 1;
            // Debug.Log("Stage: " + _board._stage);

            if (_board._stage == GameStage.Move || decisionfreqcounter > (_decisionRequestFrequency/0.02))
            {
                Debug.Log("Requesting decision");
                Debug.Log("Decision freq: " + decisionfreqcounter);
                RequestDecision();
                decisionfreqcounter = 0;
            }
            decisionfreqcounter += 1;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            sensor.AddObservation(_feedback);
            _feedback = 0;
            // sensor.AddObservation(running_feedback / running_feedback_count); // use accumulated
            running_feedback = 0;
            running_feedback_count = 0;
            sensor.AddObservation(Time.realtimeSinceStartup);
            sensor.AddObservation(trajID);
            sensor.AddObservation(_board.vector_state);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            var a1 = actions.ContinuousActions[0];
            var a2 = actions.ContinuousActions[1];
            var a3 = actions.ContinuousActions[2];
            var action = new Vector3(a1, a2, a3);

            // ActionSegment<int> actSegment = actions.DiscreteActions;
            // var action = (AIAction)actSegment[0];
            Debug.Log("Action received: " + action);
            if (action == new Vector3(0, 0, 0))
            {
                return;
            }
            ExecuteAction(action);
            _lastAction = action;

            if (_isDone)
            {
                EndEpisode();
                _isDone = false;
                trajID++;
            }
        }

        private void ExecuteAction(Vector3 action)
        {
            // _board.HandleClientControl(NetCommand.Up);
            _board.ExecuteActionVector(action);
            // switch (action)
            // {
            //     case AIAction.UP:
            //         _board.HandleClientControl(NetCommand.Up);
            //         break;

            //     case AIAction.DOWN:
            //         _board.HandleClientControl(NetCommand.Down);
            //         break;

            //     case AIAction.SHOOT:
            //         _board.HandleClientControl(NetCommand.Shoot);
            //         break;

            //     case AIAction.NO_MOVE:
            //     default:
            //         break;
            // }
        }

        private void DecisionRequestLoop()
        {
            RequestDecision();
        }

        private void OnFrameEnded(int frameCount, int score)
        {
            
            // AddReward(score);
        }

        private void OnGameEnded()
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