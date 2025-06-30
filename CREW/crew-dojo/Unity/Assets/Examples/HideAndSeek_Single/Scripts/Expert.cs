using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.AI;


namespace Examples.HideAndSeek_Single
{
    public class Expert : MonoBehaviour
    {
        public GameManager gameManager;
        public NavMeshAgent agent;

        private Vector3 last_position;

        string textpath = "/home/lingyu/Desktop/action_path.txt";
        // public Transform goal;
        // Start is called before the first frame update

        void Start()
        {

            gameManager = FindObjectOfType<GameManager>();
            agent = GetComponent<NavMeshAgent>();
            // agent.destination = gameManager._treasure.transform.position;
            Debug.Log($"Treasure: {gameManager._treasure.transform.position}");
            agent.enabled = true;


            // ResetNavMeshAgent();
            // StartCoroutine(WaitAndSetDestination());
            agent.SetDestination(gameManager._treasure.transform.position);

            // agent.destination = new Vector3(0, 0, 0);
        }

        void FixedUpdate()
        {
            Debug.Log($"{agent.hasPath}");
            Debug.Log($"On Nav Mesh: {agent.isOnNavMesh}");
            if (!agent.hasPath)
            {
                agent.ResetPath();
                agent.SetDestination(gameManager._treasure.transform.position);
            }


            // File.AppendAllText(textpath, $"{agent.nextPosition - last_position}\n");
            // Debug.Log($"{agent.nextPosition - last_position}");
            // Debug.Log($"{agent.transform.position}");
            // Debug.Log($"{agent.transform.rotation.eulerAngles}");
            last_position = agent.nextPosition;


            // agent.destination = gameManager._treasure.transform.position;

        }

        // public void ResetNavMeshAgent()
        // {
        //     Debug.Log("Resetting NavMeshAgent");
        //     // agent.ResetPath();
        //     agent.SetDestination(gameManager._treasure.transform.position);
        // }

        public IEnumerator WaitAndSetDestination(Vector2 des)
        {
            yield return null; // waits one frame
            agent.ResetPath();
            agent.SetDestination(des);
            //agent.SetDestination(gameManager._treasure.transform.position);
        }
        // agent.destination = gameManager._treasure.transform.position;

        // agent.destination = goal.position;
    }

}