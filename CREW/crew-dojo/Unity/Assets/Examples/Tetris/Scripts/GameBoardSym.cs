using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using Nakama.TinyJson;

namespace Examples.Tetris
{
    // used for AI simulation
    public class GameBoardSym
    {
        private readonly bool[,] _board; // true means occupied
        private readonly Vector2Int[] _controlled; // object under control
        private readonly int _controlledCenter = 1;

        public readonly List<List<int>> ActionPlans = new();
        public readonly List<string> FinalStates = new();
        public readonly string StartState;

        private readonly HashSet<ulong> _visitedState = new();

        public GameBoardSym(TetrisUnit[,] states, List<Vector2Int> controlled)
        {
            var cols = states.GetLength(0);
            var rows = states.GetLength(1);
            _board = new bool[cols, rows];
            for (var col = 0; col < cols; ++col)
            {
                for (var row = 0; row < rows; ++row)
                {
                    var obj = states[col, row];
                    _board[col, row] = obj.State == TetrisState.Free;
                }
            }

            Debug.Assert(controlled.Count > 0);
            _controlled = controlled.ToArray();

            StartState = EncodeState(_board, _controlled);
        }

        public void Simulate(bool[,] board = null, Vector2Int[] controlled = null, List<int> actions = null)
        {
            board ??= _board;
            controlled ??= _controlled;
            actions ??= new();
            var state = EncodeControlledPositions(controlled);

            if (_visitedState.Contains(state))
            {
                return;
            }

            for (var action = 0; action < (int)AIAction.Count; ++action)
            {
                var newBoard = board.Clone() as bool[,];
                var newControlled = controlled.Clone() as Vector2Int[];

                actions.Add(action);
                ExecuteAction((AIAction)action, newBoard, newControlled, out _);
                actions.Add((int)AIAction.STEP_DOWN);
                ExecuteAction(AIAction.STEP_DOWN, newBoard, newControlled, out var stopped);

                var newState = EncodeControlledPositions(newControlled);

                if (stopped && !_visitedState.Contains(newState))
                {
                    FinalStates.Add(EncodeState(newBoard, newControlled));
                    ActionPlans.Add(new(actions));
                    _visitedState.Add(newState);
                }
                else
                {
                    Simulate(newBoard, newControlled, actions);
                }

                actions.RemoveRange(actions.Count - 2, 2);
            }

            _visitedState.Add(state);
        }

        private void ExecuteAction(AIAction action, bool[,] newBoard, Vector2Int[] newControlled, out bool stopped)
        {
            var centerPos = newControlled[_controlledCenter];
            var newPositions = new List<Vector2Int>(newControlled);

            switch (action)
            {
                case AIAction.MOVE_LEFT:
                    {
                        newPositions = newPositions.Select(pos => new Vector2Int(pos.x - 1, pos.y)).ToList();
                        stopped = Overlapped(newBoard, newPositions);
                        if (stopped)
                        {
                            newPositions = new List<Vector2Int>(newControlled);
                        }
                    }
                    break;

                case AIAction.MOVE_RIGHT:
                    {
                        newPositions = newPositions.Select(pos => new Vector2Int(pos.x + 1, pos.y)).ToList();
                        stopped = Overlapped(newBoard, newPositions);
                        if (stopped)
                        {
                            newPositions = new List<Vector2Int>(newControlled);
                        }
                    }
                    break;

                case AIAction.ROTATE:
                    {
                        newPositions = newPositions.Select(pos =>
                        {
                            var offset = pos - centerPos;
                            pos = centerPos + new Vector2Int(
                                -offset.y,
                                +offset.x
                            );
                            return pos;
                        }).ToList();
                        stopped = Overlapped(newBoard, newPositions);
                        if (stopped)
                        {
                            newPositions = new List<Vector2Int>(newControlled);
                        }
                    }
                    break;

                case AIAction.FLIP:
                    {
                        newPositions = newPositions.Select(pos => new Vector2Int(centerPos.x * 2 - pos.x, pos.y)).ToList();
                        stopped = Overlapped(newBoard, newPositions);
                        if (stopped)
                        {
                            newPositions = new List<Vector2Int>(newControlled);
                        }
                    }
                    break;

                case AIAction.DROP:
                    stopped = false;
                    while (!stopped)
                    {
                        ExecuteAction(AIAction.STEP_DOWN, newBoard, newControlled, out stopped);
                    }
                    break;

                case AIAction.STEP_DOWN:
                    {
                        newPositions = newPositions.Select(pos => new Vector2Int(pos.x, pos.y + 1)).ToList();
                        stopped = Overlapped(newBoard, newPositions);
                        if (stopped)
                        {
                            newPositions = new List<Vector2Int>(newControlled);
                        }
                    }
                    break;

                case AIAction.NO_MOVE:
                default:
                    stopped = false;
                    break;
            }

            if (action != AIAction.DROP)
            {
                for (var idx = 0; idx < newControlled.Length; ++idx)
                {
                    newControlled[idx] = newPositions[idx];
                }
            }
        }

        // refer to https://www.cs.utexas.edu/~bradknox/papers/icdl08-knox.pdf
        private string EncodeState(bool[,] board, Vector2Int[] controlled)
        {
            board = board.Clone() as bool[,];
            foreach (var pos in controlled)
            {
                board[pos.x, pos.y] = true;
            }

            var features = new List<int>();
            var cols = board.GetLength(0);
            var rows = board.GetLength(1);

            // count heights and holes for each column
            var heights = new List<int>();
            var totalHoles = 0;
            for (var col = 0; col < cols; ++col)
            {
                var lastHoles = 0;
                var maxHeight = 0;
                for (var row = rows - 1; row >= 0; --row)
                {
                    var actualRow = rows - row;
                    if (board[col, row])
                    {
                        totalHoles += lastHoles;
                        lastHoles = 0;
                        maxHeight = actualRow;
                    }
                    else
                    {
                        lastHoles++;
                    }
                }
                heights.Add(maxHeight);
            }

            // add features (21 values)
            features.AddRange(heights);
            features.Add(heights.Max());
            for (var idx = 1; idx < heights.Count; ++idx)
            {
                features.Add(Math.Abs(heights[idx] - heights[idx - 1]));
            }
            features.Add(totalHoles);

            return JsonWriter.ToJson(features);
        }

        private bool Overlapped(bool[,] board, List<Vector2Int> positions)
        {
            var cols = board.GetLength(0);
            var rows = board.GetLength(1);
            return positions.Any(pos => pos.x < 0 || pos.x >= cols || pos.y < 0 || pos.y >= rows || board[pos.x, pos.y]);
        }

        private ulong EncodeControlledPositions(Vector2Int[] controlled)
        {
            var vals = controlled.OrderBy(v => v.x).ThenBy(v => v.y).ToArray();

            ulong result = 0;
            for (var idx = 0; idx < Math.Min(4, vals.Length); ++idx)
            {
                result |= (ulong)(vals[idx][0]) << (8 * (idx * 2));
                result |= (ulong)(vals[idx][1]) << (8 * (idx * 2 + 1));
            }
            return result;
        }
    }
}