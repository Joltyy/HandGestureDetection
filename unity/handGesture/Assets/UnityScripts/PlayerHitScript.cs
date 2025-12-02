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

    [Header("Tickle Settings")]
    [Tooltip("Gesture index value that represents 'tickle' in your model. Set to -1 to ignore index check.")]
    public int tickleGestureIndex = 3;
    [Tooltip("Speed (px/s) required to trigger a tickle when using the Python receiver")]
    public float tickleSpeedThreshold = 20f;
    [Tooltip("Animator trigger name for tickle")]
    public string tickleTriggerName = "tickled";
    [Tooltip("Seconds to ignore subsequent triggers after a tickle is fired")]
    public float tickleCooldown = 0.6f;

    // internal state for edge-detection and per-gesture cooldowns
    private float lastFrameSpeed = 0f;
    private float lastTriggerTimePunch = -999f;
    private float lastTriggerTimeSlap = -999f;
    private float lastTriggerTimeTickle = -999f;

    [Header("Player Animator")]
    public Animator anim;

    [Header("keybinds")]
    public InputAction punchAction;

    [Header("ScoreManager")]
    public scoreManager scoreMgr;

    [Header("Damage PopUp")]
    public DamagePopUp damagePopUpPrefab;
    public float popupVerticalOffset = 2f;
    public Transform popupParent;

    void Start()
    {
        punchAction.Enable();
    }

    void Update()
    {
        // thresholds per gesture
        float punchThreshold = speedThreshold;
        float slapThreshold = slapSpeedThreshold;
        float tickleThreshold = tickleSpeedThreshold;

        // manual punch key still works
        bool manualPunch = punchAction.WasPressedThisFrame();
        if (manualPunch)
        {
            if (anim != null)
                anim.SetTrigger("punched");
            lastTriggerTimePunch = Time.time;
            // update lastFrameSpeed conservatively
            lastFrameSpeed = Mathf.Max(lastFrameSpeed, 0f);
            return;
        }

        if (pyReciever == null)
            return;

        int remoteIndex;
        float remoteSpeed;
        string remoteLastUpdate;
        pyReciever.GetLatest(out remoteIndex, out remoteSpeed, out remoteLastUpdate);

        bool handledThisFrame = false;

        // Punch detection (rising edge)
        bool punchIndexOk = (punchGestureIndex < 0) || (remoteIndex == punchGestureIndex);
        bool punchRising = (lastFrameSpeed < punchThreshold) && (remoteSpeed >= punchThreshold);
        if (!handledThisFrame && punchRising && ((Time.time - lastTriggerTimePunch) >= triggerCooldown) && punchIndexOk)
        {
            Debug.Log($"Punch detected: speed={remoteSpeed:F2} (threshold {punchThreshold}) (last update: {remoteLastUpdate})");
            if (anim != null)
                anim.SetTrigger("punched");
            if (scoreMgr != null)
            {
                int added = scoreMgr.AddGestureScore("punch", remoteSpeed);
                TryShowPopup(added);
            }
            lastTriggerTimePunch = Time.time;
            handledThisFrame = true;
        }

        // Slap detection
        bool slapIndexOk = (slapGestureIndex < 0) || (remoteIndex == slapGestureIndex);
        bool slapRising = (lastFrameSpeed < slapThreshold) && (remoteSpeed >= slapThreshold);
        // Always log when motion surpasses slap threshold (helpful while animations aren't set)
        if (slapRising)
        {
            Debug.Log($"[Motion] Slap speed threshold exceeded: speed={remoteSpeed:F2} (threshold {slapThreshold}) predictedIndex={remoteIndex} (last update: {remoteLastUpdate})");
        }

        if (!handledThisFrame && slapRising && ((Time.time - lastTriggerTimeSlap) >= triggerCooldown) && slapIndexOk)
        {
            float strength = Mathf.Clamp(remoteSpeed * slapSpeedToStrength, 0f, 1f);
            Debug.Log($"Slap detected: speed={remoteSpeed:F2} -> strength={strength:F2} (threshold {slapSpeedThreshold}) (last update: {remoteLastUpdate})");
            if (anim != null)
            {
                if (!string.IsNullOrEmpty(slapStrengthParam))
                    anim.SetFloat(slapStrengthParam, strength);
                if (!string.IsNullOrEmpty(slapTriggerName))
                    anim.SetTrigger(slapTriggerName);
            }
            lastTriggerTimeSlap = Time.time;
            handledThisFrame = true;
        }

        // Tickle detection
        bool tickleIndexOk = (tickleGestureIndex < 0) || (remoteIndex == tickleGestureIndex);
        bool tickleRising = (lastFrameSpeed < tickleThreshold) && (remoteSpeed >= tickleThreshold);
        // Always log when motion surpasses tickle threshold
        if (tickleRising)
        {
            Debug.Log($"[Motion] Tickle speed threshold exceeded: speed={remoteSpeed:F2} (threshold {tickleThreshold}) predictedIndex={remoteIndex} (last update: {remoteLastUpdate})");
        }

        if (!handledThisFrame && tickleRising && ((Time.time - lastTriggerTimeTickle) >= tickleCooldown) && tickleIndexOk)
        {
            Debug.Log($"Tickle detected: speed={remoteSpeed:F2} (threshold {tickleThreshold}) (last update: {remoteLastUpdate})");
            if (anim != null && !string.IsNullOrEmpty(tickleTriggerName))
                anim.SetTrigger(tickleTriggerName);
            lastTriggerTimeTickle = Time.time;
            handledThisFrame = true;
        }

        // update lastFrameSpeed for next-frame edge detection
        lastFrameSpeed = remoteSpeed;
    }

    //show damage popup
    void TryShowPopup(int amount)
    {
        if (damagePopUpPrefab == null) return;
        if (amount <= 0) return;

        Vector3 pos = transform.position + Vector3.up * popupVerticalOffset;

        if (popupParent != null)
        {
            var inst = Instantiate(damagePopUpPrefab, popupParent);
            inst.transform.position = pos;
            inst.Init(amount);
        }
        else
        {
            // fallback: for non-UGUI (3D TextMeshPro) prefabs
            DamagePopUp.Spawn(damagePopUpPrefab, pos, amount);
        }
    }
}
