using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.Events;

public class timer : MonoBehaviour
{
    [Tooltip("Time to count down (seconds)")]
    public float timeRemaining = 60f;

    [Tooltip("Assign a UI Slider to display the remaining time (optional)")]
    public Slider timeSlider;

    [Tooltip("Optional TextMeshProUGUI to display numeric time")]
    public TextMeshProUGUI timeLabel;

    [Tooltip("Event invoked when timer reaches zero (optional)")]
    public UnityEvent onTimerEnd;

    [Header("Gradient Colors")]
    public Color fullColor = Color.green;
    public Color midColor = Color.yellow;
    public Color lowColor = Color.red;

    [Header("Handle (knob) behavior")]
    [Range(0f, 1f)]
    public float handleShowThreshold = 0.2f;
    public float handlePulseSpeed = 6f;
    public float handlePulseAmount = 0.15f;

    [Header("Flare (appears when low)")]
    public GameObject flareObject;
    [Range(0f, 1f)]
    public float flareThreshold = 0.2f;
    public float flareMaxScale = 1.6f;
    public float flareSpeed = 4f;

    [Tooltip("Show centiseconds in the label (2 digits). Display format: mm:ss:cc")]
    public bool showCentiseconds = true;

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
        // Slider setup
        if (timeSlider != null)
        {
            timeSlider.maxValue = Mathf.Max(0.0001f, timeRemaining);
            timeSlider.value = timeRemaining;
            timeSlider.interactable = false;

            duration = timeSlider.maxValue;
            if (timeSlider.fillRect != null)
                sliderFillImage = timeSlider.fillRect.GetComponent<Image>();

            if (timeSlider.handleRect != null)
            {
                handleGameObject = timeSlider.handleRect.gameObject;
                handleImage = handleGameObject.GetComponent<Image>();
                handleBaseScale = handleGameObject.transform.localScale;
            }
        }
        else
        {
            duration = timeRemaining;
        }

        // Label: no styling, no re-parenting, leave as placed in the scene
        if (timeLabel != null)
        {
            timeLabel.raycastTarget = false;
        }

        // Flare setup
        if (flareObject != null)
        {
            flareBaseScale = flareObject.transform.localScale;
            flareImage = flareObject.GetComponent<Image>();
            flareParticles = flareObject.GetComponentInChildren<ParticleSystem>(true);
            flareObject.SetActive(false);
        }

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
            if (timeSlider.maxValue < time)
            {
                timeSlider.maxValue = time;
                duration = timeSlider.maxValue;
            }
            timeSlider.value = time;
        }

        float t = (duration > 0f) ? Mathf.Clamp01(time / duration) : 0f;

        // gradient color
        Color currentColor;
        if (t >= 0.5f)
        {
            float u = (t - 0.5f) / 0.5f;
            currentColor = Color.Lerp(midColor, fullColor, u);
        }
        else
        {
            float u = t / 0.5f;
            currentColor = Color.Lerp(lowColor, midColor, u);
        }

        if (sliderFillImage != null)
            sliderFillImage.color = currentColor;

        // Simple label text only
        if (timeLabel != null)
        {
            int totalSeconds = Mathf.FloorToInt(time);
            int minutes = totalSeconds / 60;
            int seconds = totalSeconds % 60;
            int centis = Mathf.FloorToInt((time - totalSeconds) * 100f);

            if (showCentiseconds)
                timeLabel.text = string.Format("{0:00}:{1:00}:{2:00}", minutes, seconds, centis);
            else
                timeLabel.text = string.Format("{0:00}:{1:00}", minutes, seconds);

            timeLabel.color = currentColor;
        }

        // Handle logic
        bool showHandle = (t <= handleShowThreshold);
        if (handleGameObject != null)
        {
            if (showHandle)
            {
                if (!handleGameObject.activeSelf) handleGameObject.SetActive(true);
                float pulse = 1f + handlePulseAmount * Mathf.Sin(Time.time * handlePulseSpeed);
                handleGameObject.transform.localScale = handleBaseScale * pulse;
                if (handleImage != null) handleImage.color = currentColor;
            }
            else
            {
                if (handleGameObject.activeSelf) handleGameObject.SetActive(false);
            }
        }

        // Flare logic
        bool shouldFlare = (t <= flareThreshold);
        if (flareObject != null)
        {
            if (shouldFlare)
            {
                if (!flareObject.activeSelf) flareObject.SetActive(true);
                float pulse = 0.5f + 0.5f * Mathf.Sin(Time.time * flareSpeed);
                float scaleMul = Mathf.Lerp(1f, flareMaxScale, pulse);
                flareObject.transform.localScale = flareBaseScale * scaleMul;
                if (flareImage != null) flareImage.color = currentColor;
                if (flareParticles != null && !flareParticles.isPlaying) flareParticles.Play(true);
            }
            else
            {
                if (flareObject.activeSelf) flareObject.SetActive(false);
                if (flareParticles != null && flareParticles.isPlaying)
                    flareParticles.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            }
        }
    }

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
