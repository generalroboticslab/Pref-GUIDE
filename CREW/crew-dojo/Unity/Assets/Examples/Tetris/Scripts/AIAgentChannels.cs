using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.SideChannels;
using Nakama.TinyJson;
using Dojo;

namespace Examples.Tetris
{
    public class EventChannel : SideChannel
    {
        private const string LOGSCOPE = "EventChannel";

        private readonly DojoConnection _connection;
        private readonly GameBoard _board;

        public readonly bool IsInitialized;

        public EventChannel(DojoConnection connection, GameBoard board)
        {
            IsInitialized = false;
            var args = Environment.GetCommandLineArgs();

            for (var idx = 0; idx < args.Length; ++idx)
            {
                var arg = args[idx];
                if (arg.Equals("-EventChannelID") && idx < args.Length - 1)
                {
                    ChannelId = new Guid(args[idx + 1]);
                    Debug.Log($"ChannelID: {ChannelId}");
                    IsInitialized = true;
                    break;
                }
            }

            _connection = connection;
            _board = board;
            if (IsInitialized)
            {
                //_board.OnGameOver += OnGameOverEvent;
                //_board.OnScoreUpdate += OnGameScoreUpdate;
                _board.OnNewObjectSpawned += OnNewObjectSpawned;
            }
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {

        }

        private void OnNewObjectSpawned()
        {
            var newEvent = new List<object>() { "E", "ObjectSpawned" };

            using (var msgOut = new OutgoingMessage())
            {
                msgOut.WriteString(JsonWriter.ToJson(newEvent));
                QueueMessageToSend(msgOut);
            }

            Debug.Log($"{LOGSCOPE}: OnNewObjectSpawned");
        }
    }

    /// <summary>
    /// Construct a communication channel and handles time toggling
    /// </summary>
    public class ToggleTimestepChannel : SideChannel
    {
        private const string LOGSCOPE = "ToggleTimestepChannel";

        private readonly GameBoard _board;

        public readonly bool IsInitialized;

        public ToggleTimestepChannel(GameBoard board)
        {
            IsInitialized = false;
            var args = Environment.GetCommandLineArgs();

            for (var idx = 0; idx < args.Length; ++idx)
            {
                var arg = args[idx];
                if (arg.Equals("-ToggleTimestepChannelID") && idx < args.Length - 1)
                {
                    ChannelId = new Guid(args[idx + 1]);
                    Debug.Log($"ChannelID: {ChannelId}");
                    IsInitialized = true;
                    break;
                }
            }

            _board = board;
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {
            if (IsInitialized)
            {
                _board.CallToggleTimestep();
                Debug.Log($"{LOGSCOPE}: OnMessageReceived");
            }
        }
    }
}
