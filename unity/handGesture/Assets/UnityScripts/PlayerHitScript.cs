using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.InputSystem;

public class PlayerHitScript : MonoBehaviour
{
    [Header("Python Receiver Script")]
    public pythonreciever pyReciever;

    [Header("Player Animator")]
    public Animator anim;

    [Header("keybinds")]
    public InputAction punchAction;

    void Start()
    {
        punchAction.Enable();
    }

    void Update()
    {
        if (punchAction.WasPressedThisFrame() || pyReciever.gestureSpeed >= 100) //testing animation, press 1 to trigger punch
        {
            anim.SetTrigger("punched");
        }
    }
}
