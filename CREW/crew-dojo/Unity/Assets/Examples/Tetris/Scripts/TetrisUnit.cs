using UnityEngine;

namespace Examples.Tetris
{
    [ExecuteInEditMode]
    [RequireComponent(typeof(Renderer))]
    public class TetrisUnit : MonoBehaviour
    {
        [SerializeField]
        TetrisBoard _board;

        [SerializeField]
        Vector2Int _boardPosition = new(0, 0);

        [SerializeField]
        Material _materialDefault;

        [SerializeField]
        Material _materialControlled;

        private Color _color = Color.white;

        private void OnValidate()
        {
            UpdatePosition();
        }

        private void UpdatePosition()
        {
            if (_board != null)
            {
                var gridX = _board.BoardSize.x;
                var gridY = _board.BoardSize.y;
                var halfX = (gridX - 1) * 0.5f;
                var halfY = (gridY - 1) * 0.5f;
                _boardPosition.x = Mathf.Clamp(_boardPosition.x, 0, gridX - 1);
                _boardPosition.y = Mathf.Clamp(_boardPosition.y, 0, gridY - 1);
                transform.localPosition = new Vector3(_boardPosition.x - halfX, halfY - _boardPosition.y, -1.0f);
            }
        }

        public TetrisBoard Board
        {
            get
            {
                return _board;
            }
            set
            {
                _board = value;
            }
        }

        public Vector2Int BoardPosition
        {
            get
            {
                return _boardPosition;
            }
            set
            {
                _boardPosition = value;
                UpdatePosition();
            }
        }

        public Color UnitColor
        {
            get
            {
                return _color;
            }
            set
            {
                GetComponent<Renderer>().material.color = value;
                _color = value;
            }
        }

        private TetrisState _state = TetrisState.Free;
        public TetrisState State
        {
            get
            {
                return _state;
            }
            set
            {
                if (_state != value)
                {
                    var renderer = GetComponent<Renderer>();
                    var color = renderer.material.color;
                    _state = value;
                    if (_state == TetrisState.InControl)
                    {
                        renderer.material = _materialControlled;
                        UnitColor = color;
                    }
                    else
                    {
                        renderer.material = _materialDefault;
                        UnitColor = color;
                    }
                    if (_state == TetrisState.Unoccupied)
                    {
                        gameObject.SetActive(false);
                    }
                    else
                    {
                        gameObject.SetActive(true);
                    }
                }
            }
        }
    }
}
