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
    public float slapSpeedThreshold = 100f;
    [Tooltip("Multiplier to convert px/s to animator strength (0-1). Adjust to fit your animation.")]
    public float slapSpeedToStrength = 1f / 300f;
    [Tooltip("Animator trigger name for slap")]
    public string slapTriggerName = "isSlap";
    [Tooltip("Animator float parameter name to set slap strength")]
    public string slapStrengthParam = "slapStrength";

    [Header("Tickle Settings")]
    [Tooltip("Gesture index value that represents 'tickle' in your model. Set to -1 to ignore index check.")]
    public int tickleGestureIndex = 3;
    [Tooltip("Speed (px/s) required to keep tickle active (>= to be true)")]
    public float tickleSpeedThreshold = 100f;
    [Tooltip("Animator bool parameter name for tickle state")]
    public string tickleBoolName = "isTickle";
    [Tooltip("Points per second added while tickle is active")]
    public float ticklePointsPerSecond = 2f;

    // internal state for edge-detection and per-gesture cooldowns
    private float lastFrameSpeed = 0f;
    private float lastTriggerTimePunch = -999f;
    private float lastTriggerTimeSlap = -999f;
    private float lastTriggerTimeTickle = -999f; // unused for continuous, kept for compatibility
    private float _ticklePointsAccumulator = 0f;

    [Header("Player Animator")]
    public Animator anim;

    [Header("keybinds")]
    public InputAction punchAction;

    [Header("ScoreManager")]
    public scoreManager scoreMgr;

    [Header("Animator Parameters")]
    [Tooltip("Animator trigger parameter for punch")]
    public string punchTriggerName = "isPunch";

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
            if (anim != null && !string.IsNullOrEmpty(punchTriggerName))
                anim.SetTrigger(punchTriggerName);
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
            if (anim != null && !string.IsNullOrEmpty(punchTriggerName))
                anim.SetTrigger(punchTriggerName);
            if (scoreMgr != null)
            {
                scoreMgr.AddGestureScore("punch", remoteSpeed);
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
            Debug.Log($"Slap detected: speed={remoteSpeed:F2} (threshold {slapSpeedThreshold}) (last update: {remoteLastUpdate})");
            if (anim != null && !string.IsNullOrEmpty(slapTriggerName))
                anim.SetTrigger(slapTriggerName);
            lastTriggerTimeSlap = Time.time;
            if (scoreMgr != null)
            {
                scoreMgr.AddGestureScore("slap", remoteSpeed);
            }
            handledThisFrame = true;
        }

        // Tickle: continuous boolean state + slow continuous scoring
        bool tickleIndexOk = (tickleGestureIndex < 0) || (remoteIndex == tickleGestureIndex);
        bool tickleActive = tickleIndexOk && (remoteSpeed >= tickleThreshold);
        if (anim != null && !string.IsNullOrEmpty(tickleBoolName))
            anim.SetBool(tickleBoolName, tickleActive);
        if (tickleActive && scoreMgr != null)
        {
            _ticklePointsAccumulator += ticklePointsPerSecond * Time.deltaTime;
            if (_ticklePointsAccumulator >= 1f)
            {
                int grant = Mathf.FloorToInt(_ticklePointsAccumulator);
                _ticklePointsAccumulator -= grant;
                if (grant > 0)
                    scoreMgr.AddPoints(grant);
            }
        }
        else
        {
            _ticklePointsAccumulator = 0f;
        }

        // update lastFrameSpeed for next-frame edge detection
        lastFrameSpeed = remoteSpeed;
    }
}
