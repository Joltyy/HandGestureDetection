using UnityEngine;

public class testanim : MonoBehaviour
{

    public Animator anim;
    public pythonreciever pyReceiver;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        if (anim != null)
            anim.Play("Idle");
    }

    // Update is called once per frame
    void Update()
    {
        if (pyReceiver != null && pyReceiver.gestureIndex == 1)
        {
            if (anim != null)
                anim.SetTrigger("HitandFall");
        }
    }
}
