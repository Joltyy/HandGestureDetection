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

    [Header("Time Label Styling")]
    [Tooltip("Use TMP rich text formatting to present minutes and seconds with different sizes")]
    public bool useRichTextStyle = true;
    [Tooltip("Main font size for minutes (px) when using rich style")]
    public int mainFontSize = 56;
    [Tooltip("Font size for seconds (px) when using rich style")]
    public int secondsFontSize = 40;
    [Tooltip("Enable a simple UI Outline component on the TMP label")]
    public bool useOutline = true;
    [Tooltip("Outline color")]
    public Color outlineColor = Color.black;
    [Tooltip("Outline thickness (effectDistance in px)")]
    public Vector2 outlineDistance = new Vector2(2f, -2f);
    [Tooltip("Enable a simple UI Shadow component on the TMP label")]
    public bool useShadow = true;
    [Tooltip("Shadow color")]
    public Color shadowColor = new Color(0f, 0f, 0f, 0.5f);
    [Tooltip("Shadow distance in px")]
    public Vector2 shadowDistance = new Vector2(2f, -2f);

    [Tooltip("Optional Image placed behind the text to improve legibility (assign a UI Image)")]
    public Image timeLabelBackground;
    [Tooltip("Background color (alpha controls opacity)")]
    public Color backgroundColor = new Color(0f, 0f, 0f, 0.45f);
    [Tooltip("Padding (x=horizontal, y=vertical) added to background around the text in px")]
    public Vector2 backgroundPadding = new Vector2(18f, 8f);

    [Tooltip("If true will try to parent/timeLabel to the slider's fillRect so it's visually inside the fill")]
    public bool placeLabelInsideFill = true;
    [Tooltip("If true places the time label above the slider instead of inside it")]
    public bool placeLabelAboveSlider = false;
    [Tooltip("Vertical offset in pixels when placing label above the slider")]
    public float labelAboveOffsetY = 8f;
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

    // internal references for styling components
    Outline tmpOutline;
    Shadow tmpShadow;

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
            if (placeLabelInsideFill && timeSlider.fillRect != null && timeLabel != null && !placeLabelAboveSlider)
            {
                timeLabel.rectTransform.SetParent(timeSlider.fillRect, false);
            }
        }
        else
        {
            duration = timeRemaining;
        }

        // ensure the label is centered in its parent (so it appears "inside" the slider) or positioned above the slider
        if (timeLabel != null)
        {
            var rt = timeLabel.rectTransform;
            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.pivot = new Vector2(0.5f, 0.5f);
            rt.anchoredPosition = Vector2.zero;
            timeLabel.raycastTarget = false; // avoid blocking UI

            // TMP styling setup
            timeLabel.richText = true;
            timeLabel.enableAutoSizing = false;
            timeLabel.fontSize = mainFontSize;
            timeLabel.alignment = TextAlignmentOptions.Center;
            timeLabel.fontStyle = FontStyles.Bold;

            // add or configure Outline component
            if (useOutline)
            {
                tmpOutline = timeLabel.gameObject.GetComponent<Outline>();
                if (tmpOutline == null) tmpOutline = timeLabel.gameObject.AddComponent<Outline>();
                tmpOutline.effectColor = outlineColor;
                tmpOutline.effectDistance = outlineDistance;
            }
            else
            {
                tmpOutline = timeLabel.gameObject.GetComponent<Outline>();
                if (tmpOutline != null) Destroy(tmpOutline);
            }

            // add or configure Shadow component
            if (useShadow)
            {
                tmpShadow = timeLabel.gameObject.GetComponent<Shadow>();
                if (tmpShadow == null) tmpShadow = timeLabel.gameObject.AddComponent<Shadow>();
                tmpShadow.effectColor = shadowColor;
                tmpShadow.effectDistance = shadowDistance;
            }
            else
            {
                tmpShadow = timeLabel.gameObject.GetComponent<Shadow>();
                if (tmpShadow != null) Destroy(tmpShadow);
            }

            // background image setup
            if (timeLabelBackground != null)
            {
                timeLabelBackground.color = backgroundColor;
                // ensure background is behind the text
                timeLabelBackground.rectTransform.SetAsFirstSibling();
                // initial sizing will be handled in UpdateDisplay
            }
        }

        // If requested, place the label above the slider (reparent to slider's parent so it sits above)
        if (placeLabelAboveSlider && timeSlider != null && timeLabel != null)
        {
            var sliderRT = timeSlider.GetComponent<RectTransform>();
            var labelRT = timeLabel.rectTransform;
            // parent to the same parent as slider so it's not clipped inside fill
            labelRT.SetParent(sliderRT.parent, false);
            // compute offset: place above slider by half slider height + offset
            float sliderHalf = sliderRT.rect.height * 0.5f;
            float labelHalf = labelRT.rect.height * 0.5f;
            labelRT.anchoredPosition = new Vector2(sliderRT.anchoredPosition.x, sliderRT.anchoredPosition.y + sliderHalf + labelHalf + labelAboveOffsetY);
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
            int totalSeconds = Mathf.FloorToInt(time);
            int minutes = totalSeconds / 60;
            int seconds = totalSeconds % 60;
            int centis = Mathf.FloorToInt((time - totalSeconds) * 100f);

            if (showCentiseconds)
            {
                // single-size display including centiseconds (mm:ss:cc)
                timeLabel.text = string.Format("{0:00}:{1:00}:{2:00}", minutes, seconds, centis);
                timeLabel.fontSize = mainFontSize;
            }
            else if (useRichTextStyle)
            {
                // mixed sizes for minutes and seconds using TMP rich text
                int mSize = Mathf.Max(1, mainFontSize);
                int sSize = Mathf.Max(1, secondsFontSize);
                timeLabel.text = $"<size={mSize}>{minutes:00}</size><size={sSize}>:{seconds:00}</size>";
            }
            else
            {
                // same-size minutes and seconds
                timeLabel.text = string.Format("{0:00}:{1:00}", minutes, seconds);
                timeLabel.fontSize = mainFontSize;
            }

            // tint label to match current fill color (keeps contrast)
            timeLabel.color = currentColor;

            // adjust background to fit text
            if (timeLabelBackground != null)
            {
                // request preferred values from TMP for current text
                Vector2 pref = timeLabel.GetPreferredValues(timeLabel.text);
                Vector2 bgSize = new Vector2(pref.x + backgroundPadding.x * 2f, pref.y + backgroundPadding.y * 2f);
                timeLabelBackground.rectTransform.sizeDelta = bgSize;
            }
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
