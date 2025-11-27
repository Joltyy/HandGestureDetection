using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.InputSystem;

public class PlayerHitScript : MonoBehaviour
{
    [Header("Python Receiver Script")]
    public pythonreciever pyReciever;

    [Header("Gesture Thresholds")]
    [Tooltip("Speed (px/s) required to trigger a punch when using the Python receiver")]
    public float speedThreshold = 100f;
    [Tooltip("Seconds to ignore subsequent triggers after a punch is fired")]
    public float triggerCooldown = 0.5f;
    [Tooltip("Gesture index value that represents 'punch' in your model. Set to -1 to ignore index check.")]
    public int punchGestureIndex = 1;
    [Header("Slap Settings")]
    [Tooltip("Gesture index value that represents 'slap' in your model. Set to -1 to ignore index check.")]
    public int slapGestureIndex = 2;
    [Tooltip("Speed (px/s) required to trigger a slap when using the Python receiver")]
    public float slapSpeedThreshold = 80f;
    [Tooltip("Multiplier to convert px/s to animator strength (0-1). Adjust to fit your animation.")]
    public float slapSpeedToStrength = 1f / 300f;
    [Tooltip("Animator trigger name for slap")]
    public string slapTriggerName = "slapped";
    [Tooltip("Animator float parameter name to set slap strength")]
    public string slapStrengthParam = "slapStrength";

    // internal state for edge-detection and cooldown
    private float lastFrameSpeed = 0f;
    private float lastTriggerTime = -999f;

    [Header("Player Animator")]
    public Animator anim;

    [Header("keybinds")]
    public InputAction punchAction;

    void Start()
    {
        punchAction.Enable();
        if (pyReciever == null)
        {
            pyReciever = FindObjectOfType<pythonreciever>();
            if (pyReciever == null)
            {
                Debug.LogWarning("pythonreciever not found in scene — assign it in the inspector or run the receiver.");
            }
            else
            {
                Debug.Log($"Auto-assigned pythonreciever from scene. Receiver GameObject: {pyReciever.gameObject.name}");
                Debug.Log($"Receiver last update: {pyReciever.lastUpdateUtc}");
            }
        }
    }

    void Update()
    {
        bool triggered = false;
        if (punchAction.WasPressedThisFrame())
        {
            triggered = true;
        }
        else if (pyReciever != null)
        {
            // read snapshot safely from the receiver
            int remoteIndex;
            float remoteSpeed;
            string remoteLastUpdate;
            pyReciever.GetLatest(out remoteIndex, out remoteSpeed, out remoteLastUpdate);

            // edge detection: trigger only when speed rises past thresholds (prevent multiple triggers)
            bool cooldownOk = (Time.time - lastTriggerTime) >= triggerCooldown;

            // Check for punch (existing behavior)
            // bool punchIndexOk = (punchGestureIndex < 0) || (remoteIndex == punchGestureIndex);
            // bool punchRising = (lastFrameSpeed < speedThreshold) && (remoteSpeed >= speedThreshold);
            // if (punchRising && cooldownOk && punchIndexOk)
            // {
            //     Debug.Log($"Gesture speed {remoteSpeed:F2} >= {speedThreshold} (rising edge) — triggering punch (last update: {remoteLastUpdate})");
            //     triggered = true;
            //     lastTriggerTime = Time.time;
            //     // set animator trigger for punch below when applying animation
            // }

            // Check for slap: use its own index/threshold
            bool slapIndexOk = (slapGestureIndex < 0) || (remoteIndex == slapGestureIndex);
            bool slapRising = (lastFrameSpeed < slapSpeedThreshold) && (remoteSpeed >= slapSpeedThreshold);
            if (!triggered && slapRising && cooldownOk && slapIndexOk)
            {
                // compute normalized strength for animator
                float strength = Mathf.Clamp(remoteSpeed * slapSpeedToStrength, 0f, 1f);
                Debug.Log($"Slap detected: speed={remoteSpeed:F2} -> strength={strength:F2} (threshold {slapSpeedThreshold}) (last update: {remoteLastUpdate})");
                // set animator params and trigger
                if (anim != null)
                {
                    if (!string.IsNullOrEmpty(slapStrengthParam))
                        anim.SetFloat(slapStrengthParam, strength);
                    if (!string.IsNullOrEmpty(slapTriggerName))
                        anim.SetTrigger(slapTriggerName);
                }
                lastTriggerTime = Time.time;
                // mark handled so we don't also trigger punch
                triggered = false; // slap uses its own animation, don't set punch trigger
            }

            // update lastFrameSpeed for next-frame edge detection
            lastFrameSpeed = remoteSpeed;
        }

        if (triggered)
        {
            if (anim != null)
                anim.SetTrigger("punched");
        }
    }
}
