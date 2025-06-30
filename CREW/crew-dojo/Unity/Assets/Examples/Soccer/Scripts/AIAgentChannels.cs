using System;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;
using UnityEngine;
using Nakama.TinyJson;
using Dojo;

namespace Examples.Soccer
{
    public class EventChannel : SideChannel
    {
        private const string LOGSCOPE = "EventChannel";

        private readonly DojoConnection _connection;

        public readonly bool IsInitialized;

        public EventChannel(GameManager manager, DojoConnection connection)
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
            if (IsInitialized)
            {
                manager.OnGameStarted += OnGameStarted;
                manager.OnGameEnded += OnGameEnded;
                manager.OnGameScored += OnGameScored;
            }
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {

        }

        private void OnGameStarted()
        {
            // "E" for event
            var message = new List<object>() { "E", "GameStarted" };

            // send feedback
            using (var msgOut = new OutgoingMessage())
            {
                msgOut.WriteString(JsonWriter.ToJson(message));
                QueueMessageToSend(msgOut);
            }

            Debug.Log($"{LOGSCOPE}: OnGameStarted");
        }

        private void OnGameEnded()
        {
            // "E" for event
            var message = new List<object>() { "E", "GameEnded" };

            // send feedback
            using (var msgOut = new OutgoingMessage())
            {
                msgOut.WriteString(JsonWriter.ToJson(message));
                QueueMessageToSend(msgOut);
            }

            Debug.Log($"{LOGSCOPE}: OnGameEnded");
        }

        private void OnGameScored(bool isBlue)
        {
            // "E" for event
            var message = new List<object>() { "E", "GameScored", isBlue ? "blue" : "purple" };

            // send feedback
            using (var msgOut = new OutgoingMessage())
            {
                msgOut.WriteString(JsonWriter.ToJson(message));
                QueueMessageToSend(msgOut);
            }

            Debug.Log($"{LOGSCOPE}: OnGameScored");
        }
    }
}