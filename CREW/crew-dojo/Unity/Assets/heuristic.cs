using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class heuristic : MonoBehaviour
{

    [SerializeField]
    public Rigidbody rb;

    [SerializeField]
    public int speed;

    [SerializeField]
    public int hider_detect_range;

    [SerializeField]
    public int obstacledetectrange;

    Vector3 lastDirection;

    Vector3 Direction;

    public List<GameObject> target_list = new List<GameObject>();

    public List<GameObject> wall_list = new List<GameObject>();
    
    void Start()
    {
        // transform.position = new Vector3(transform.position.x, 0.5f,transform.position.z);
    }

    void Update()
    {

        target_list.Clear();
        RaycastHit[] hits1 = Physics.SphereCastAll(new Ray(transform.position, Vector3.up),1000); //figure why 5 doesn't work
        foreach (var hit in hits1)
        {
            if (hit.collider.CompareTag("Player") && !target_list.Contains(hit.collider.gameObject))
            {
                target_list.Add(hit.collider.gameObject);
            }
        }


        transform.position = transform.position +transform.forward*speed*Time.deltaTime; 
    }
}
