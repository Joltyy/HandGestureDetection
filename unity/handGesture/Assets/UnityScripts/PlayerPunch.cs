using UnityEngine;

public class PunchReact : MonoBehaviour
{
    public Animator animator;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.P))
        {
            animator.SetTrigger("isPunch");
        }
    }
}
