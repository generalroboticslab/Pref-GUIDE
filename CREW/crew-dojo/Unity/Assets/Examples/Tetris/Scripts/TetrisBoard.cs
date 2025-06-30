using UnityEngine;

namespace Examples.Tetris
{
    public class TetrisBoard : MonoBehaviour
    {
        public Vector2Int BoardSize = new(10, 20);

        [Tooltip("Where game over should be detected")]
        [Range(1, 100)]
        public int GameOverMargin = 3;

        private Material _material, _materialMargin;

        private void Awake()
        {
            UpdateBoardSize();
            _displayMargin = transform.GetChild(4).gameObject.activeSelf;
            DisplayMargin = false;
        }

        private void OnValidate()
        {
            UpdateBoardSize();
        }

        private void UpdateBoardSize()
        {
            BoardSize.x = Mathf.Clamp(BoardSize.x, 1, 100);
            BoardSize.y = Mathf.Clamp(BoardSize.y, 1, 100);

            GameOverMargin = Mathf.Max(1, Mathf.Min(GameOverMargin, BoardSize.y));

            Debug.Assert(transform.childCount >= 5);

            // update board first
            var board = transform.GetChild(0);
            if (_material == null)
            {
                _material = board.GetComponent<Renderer>().sharedMaterial;
            }
            board.transform.localScale = new Vector3(BoardSize.x, BoardSize.y, 1.0f);
            _material.SetTextureScale("_MainTex", new Vector2(BoardSize.x, BoardSize.y));
            board.GetComponent<Renderer>().material = _material;

            // update borders
            var borderLeft = transform.GetChild(1);
            borderLeft.localScale = new Vector3(1, BoardSize.y, 1.0f);
            borderLeft.localPosition = new Vector3(-(BoardSize.x + 1) * 0.5f, 0.0f, 0.0f);
            var borderRight = transform.GetChild(2);
            borderRight.localScale = new Vector3(1, BoardSize.y, 1.0f);
            borderRight.localPosition = new Vector3((BoardSize.x + 1) * 0.5f, 0.0f, 0.0f);
            var borderBottom = transform.GetChild(3);
            borderBottom.localScale = new Vector3(BoardSize.x + 2, 1.0f, 1.0f);
            borderBottom.localPosition = new Vector3(0.0f, -(BoardSize.y + 1) * 0.5f, 0.0f);

            // update game over margin
            var margin = transform.GetChild(4);
            if (_materialMargin == null)
            {
                _materialMargin = margin.GetComponent<Renderer>().sharedMaterial;
            }
            margin.localScale = new Vector3(BoardSize.x, 1.0f, 1.0f);
            margin.localPosition = new Vector3(0.0f, BoardSize.y * 0.5f - (GameOverMargin - 0.5f), -0.1f);
            _materialMargin.SetTextureScale("_MainTex", new Vector2(BoardSize.x, 1));
            margin.GetComponent<Renderer>().material = _materialMargin;
        }

        private bool _displayMargin = true;
        public bool DisplayMargin
        {
            get
            {
                return _displayMargin;
            }
            set
            {
                if (_displayMargin != value)
                {
                    _displayMargin = value;
                    var margin = transform.GetChild(4);
                    margin.gameObject.SetActive(_displayMargin);
                }
            }
        }
    }
}