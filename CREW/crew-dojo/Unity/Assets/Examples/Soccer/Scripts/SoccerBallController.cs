using Unity.Netcode;
using UnityEngine;

namespace Examples.Soccer
{
    public class SoccerBallController : NetworkBehaviour
    {
        private GameManager _gameManager;

        private void Awake()
        {
            _gameManager = FindObjectOfType<GameManager>();
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (IsServer)
            {
                var name = collision.gameObject.name;

                if (name == "GoalBlue")
                {
                    _gameManager.HasScored(false);
                }
                else if (name == "GoalPurple")
                {
                    _gameManager.HasScored(true);
                }
            }
        }
    }
}