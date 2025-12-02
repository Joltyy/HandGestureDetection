using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.Events;

public class timer : MonoBehaviour
{
    [Tooltip("Time to count down (seconds)")]
    public float timeRemaining = 60f; // 1 minute

    [Tooltip("Assign a UI Slider to display the remaining time (optional)")]
    public Slider timeSlider;

    [Tooltip("Optional TextMeshProUGUI to display numeric time")]
    public TextMeshProUGUI timeLabel;

    [Tooltip("Event invoked when timer reaches zero (optional)")]
    public UnityEvent onTimerEnd;

    [Header("Gradient Colors")]
    [Tooltip("Color when timer is full")]
    public Color fullColor = Color.green;
    [Tooltip("Color at mid point")]
    public Color midColor = Color.yellow;
    [Tooltip("Color when almost out")]
    public Color lowColor = Color.red;

    [Header("Handle (knob) behavior")]
    [Tooltip("Normalized threshold (0..1). Handle appears when remaining/time <= this value")]
    [Range(0f, 1f)]
    public float handleShowThreshold = 0.2f;
    [Tooltip("Pulse speed of the handle when visible")]
    public float handlePulseSpeed = 6f;
    [Tooltip("Pulse scale multiplier for the handle when visible")]
    public float handlePulseAmount = 0.15f;

    [Header("Flare (appears when low)")]
    [Tooltip("Optional GameObject to use as a flare (Image, ParticleSystem parent, etc.). Assign a UI Image or prefab placed over the fill.")]
    public GameObject flareObject;
    [Tooltip("Normalized threshold (0..1). Flare activates when remaining/time <= this value")]
    [Range(0f, 1f)]
    public float flareThreshold = 0.2f;
    [Tooltip("Max flare scale multiplier")]
    public float flareMaxScale = 1.6f;
    [Tooltip("Flare pulse speed")]
    public float flareSpeed = 4f;

    [Tooltip("If true will try to parent/timeLabel to the slider's fillRect so it's visually inside the fill")]
    public bool placeLabelInsideFill = true;

    bool timerIsRunning = false;

    // internal
    Image sliderFillImage;
    float duration = 60f;

    // flare internals
    Vector3 flareBaseScale;
    Image flareImage;
    ParticleSystem flareParticles;

    // handle internals
    GameObject handleGameObject;
    Image handleImage;
    Vector3 handleBaseScale;

    void Start()
    {
        // configure slider if present
        if (timeSlider != null)
        {
            timeSlider.maxValue = Mathf.Max(0.0001f, timeRemaining);
            timeSlider.value = timeRemaining;
            timeSlider.interactable = false; // display-only

            // cache duration and fill Image
            duration = timeSlider.maxValue;
            if (timeSlider.fillRect != null)
                sliderFillImage = timeSlider.fillRect.GetComponent<Image>();

            // cache handle GameObject (default Slider has a Handle child)
            if (timeSlider.handleRect != null)
            {
                handleGameObject = timeSlider.handleRect.gameObject;
                handleImage = handleGameObject.GetComponent<Image>();
                handleBaseScale = handleGameObject.transform.localScale;
            }

            // try to parent/timeLabel to fillRect so it's visually inside the slider
            if (placeLabelInsideFill && timeSlider.fillRect != null && timeLabel != null)
            {
                timeLabel.rectTransform.SetParent(timeSlider.fillRect, false);
            }
        }
        else
        {
            duration = timeRemaining;
        }

        // ensure the label is centered in its parent (so it appears "inside" the slider)
        if (timeLabel != null)
        {
            var rt = timeLabel.rectTransform;
            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.pivot = new Vector2(0.5f, 0.5f);
            rt.anchoredPosition = Vector2.zero;
            timeLabel.raycastTarget = false; // avoid blocking UI
        }

        // prepare flare
        if (flareObject != null)
        {
            flareBaseScale = flareObject.transform.localScale;
            flareImage = flareObject.GetComponent<Image>();
            flareParticles = flareObject.GetComponentInChildren<ParticleSystem>(true);

            // start disabled
            flareObject.SetActive(false);
        }

        // initial handle visibility based on current normalized time
        float initialT = (duration > 0f) ? Mathf.Clamp01(timeRemaining / duration) : 0f;
        if (handleGameObject != null)
            handleGameObject.SetActive(initialT <= handleShowThreshold);

        timerIsRunning = true;
        UpdateDisplay(timeRemaining);
    }

    void Update()
    {
        if (!timerIsRunning) return;

        if (timeRemaining > 0f)
        {
            timeRemaining -= Time.deltaTime;
            if (timeRemaining < 0f) timeRemaining = 0f;
            UpdateDisplay(timeRemaining);
        }
        else
        {
            timerIsRunning = false;
            if (onTimerEnd != null) onTimerEnd.Invoke();
            Debug.Log("Timer finished.");
        }
    }

    void UpdateDisplay(float time)
    {
        if (timeSlider != null)
        {
            // ensure slider max stays >= current time (useful if StartTimer changes duration)
            if (timeSlider.maxValue < time)
            {
                timeSlider.maxValue = time;
                duration = timeSlider.maxValue;
            }
            timeSlider.value = time;
        }

        // compute normalized t (0..1)
        float t = (duration > 0f) ? Mathf.Clamp01(time / duration) : 0f;

        // color gradient: full (green) at t=1, mid (yellow) at t=0.5, low (red) at t=0
        Color currentColor;
        if (t >= 0.5f)
        {
            // interpolate from midColor -> fullColor as t goes 0.5..1
            float u = (t - 0.5f) / 0.5f; // 0..1
            currentColor = Color.Lerp(midColor, fullColor, u);
        }
        else
        {
            // interpolate from lowColor -> midColor as t goes 0..0.5
            float u = t / 0.5f; // 0..1
            currentColor = Color.Lerp(lowColor, midColor, u);
        }

        // apply color to slider fill image if available
        if (sliderFillImage != null)
        {
            sliderFillImage.color = currentColor;
        }

        if (timeLabel != null)
        {
            int minutes = Mathf.FloorToInt(time / 60f);
            int seconds = Mathf.FloorToInt(time % 60f);
            timeLabel.text = string.Format("{0:00}:{1:00}", minutes, seconds);
            // optional: tint label to match current fill color
            timeLabel.color = currentColor;
        }

        // handle logic: show when remaining <= handleShowThreshold, pulse and tint when visible
        bool showHandle = (t <= handleShowThreshold);
        if (handleGameObject != null)
        {
            if (showHandle)
            {
                if (!handleGameObject.activeSelf) handleGameObject.SetActive(true);

                // pulse scale
                float pulse = 1f + handlePulseAmount * Mathf.Sin(Time.time * handlePulseSpeed);
                handleGameObject.transform.localScale = handleBaseScale * pulse;

                // tint handle image if present
                if (handleImage != null)
                    handleImage.color = currentColor;
            }
            else
            {
                if (handleGameObject.activeSelf) handleGameObject.SetActive(false);
            }
        }

        // flare logic: enable when normalized remaining <= threshold
        bool shouldFlare = (t <= flareThreshold);
        if (flareObject != null)
        {
            if (shouldFlare)
            {
                if (!flareObject.activeSelf) flareObject.SetActive(true);

                // pulse scale
                float pulse = 0.5f + 0.5f * Mathf.Sin(Time.time * flareSpeed);
                float scaleMul = Mathf.Lerp(1f, flareMaxScale, pulse);
                flareObject.transform.localScale = flareBaseScale * scaleMul;

                // tint flare image if present
                if (flareImage != null)
                    flareImage.color = currentColor;

                // play particles if present
                if (flareParticles != null && !flareParticles.isPlaying)
                    flareParticles.Play(true);
            }
            else
            {
                if (flareObject.activeSelf) flareObject.SetActive(false);
                if (flareParticles != null && flareParticles.isPlaying)
                    flareParticles.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            }
        }
    }

    // Public helper to start/reset the timer from other scripts
    public void StartTimer(float seconds = 60f)
    {
        timeRemaining = seconds;
        if (timeSlider != null)
        {
            timeSlider.maxValue = Mathf.Max(0.0001f, seconds);
            timeSlider.value = seconds;
            duration = timeSlider.maxValue;
        }
        else
        {
            duration = seconds;
        }

        // update handle initial visibility after changing duration
        float t = (duration > 0f) ? Mathf.Clamp01(timeRemaining / duration) : 0f;
        if (handleGameObject != null)
            handleGameObject.SetActive(t <= handleShowThreshold);

        timerIsRunning = true;
        UpdateDisplay(timeRemaining);
    }

    public void StopTimer()
    {
        timerIsRunning = false;
    }
}
