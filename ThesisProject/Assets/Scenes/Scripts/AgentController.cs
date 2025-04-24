using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Numerics;

public class MoveToGoalAgent : Agent
{

    [SerializeField] private Transform targetTransform;
    [SerializeField] private Transform[] enemies;
    [SerializeField] private Material winMaterial;
    [SerializeField] private Material loseMaterial;
    [SerializeField] private MeshRenderer floorMeshRenderer; 

    private bool initPos = false;
    private UnityEngine.Vector3 initAgentPos;
    private UnityEngine.Vector3 initTargetPos;

    public override void OnEpisodeBegin()
    {
        if (!initPos)
        {
            initAgentPos = transform.localPosition;
            initTargetPos = targetTransform.localPosition;
            initPos = true;
        }

        transform.localPosition = initAgentPos;
        targetTransform.localPosition = initTargetPos;



        //### If using Optuna, use this declaration of numEnemies ###
       // int numEnemies = Random.Range(1, enemies.Length + 1);

        //### If training or testing, use this declaration of numEnemies ###
        int numEnemies = enemies.Length;

        for (int i = 0; i < enemies.Length; i++)
        {
            if(i < numEnemies)
            {
                //activate enemy
                enemies[i].gameObject.SetActive(true);

                //randomize spawn position
                enemies[i].localPosition = new UnityEngine.Vector3(Random.Range(-5f, +5f), 0, Random.Range(0f, -16f));
            }
            else
            {
                //deactivate enemy
                enemies[i].gameObject.SetActive(false);
            }
        }
    }


    // ### GATHER ENVIRONMENT INFORMATION ###
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        
        sensor.AddObservation(targetTransform.localPosition);

        // Set max number of enemies (to pad unused observaton space)
        int maxEnemies = 6;

        for (int i = 0; i < maxEnemies; i++)
        {
            //only add active enemies
            if(i < enemies.Length && enemies[i].gameObject.activeSelf)
            {
                sensor.AddObservation(enemies[i].localPosition);
            }
            else
            {
                sensor.AddObservation(UnityEngine.Vector3.zero);
            }
        }
    }

    // ### APPLIES ACTIONS RECEIVED FROM POLICY TO AGENT'S BEHAVIOR ###
    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];

        float moveSpeed = 5f;
        transform.localPosition += new UnityEngine.Vector3(moveX, 0, moveZ) * Time.deltaTime * moveSpeed;
    }


    //### MANUALLY CONTROLLING THE AGENT (FOR TESTING) ###
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxisRaw("Horizontal");
        continuousActions[1] = Input.GetAxisRaw("Vertical");
    }




    //### INTERACTIONS WITH ENVIRONMENT ###
    private void OnTriggerEnter(Collider other) {
        if(other.TryGetComponent<Goal>(out Goal goal)) {
            SetReward(+5f);
            floorMeshRenderer.material = winMaterial;
            EndEpisode();
        }
    }

    private void OnCollisionEnter(Collision other) {
        if(other.gameObject.TryGetComponent<PhysicalWall>(out PhysicalWall physicalWall)) {
            SetReward(-0.5f);
        }

        if(other.gameObject.TryGetComponent<Enemy>(out Enemy enemy)) {
            SetReward(-1f);
            floorMeshRenderer.material = loseMaterial;
            EndEpisode();
        }
    }
}
