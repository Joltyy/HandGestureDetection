using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class TimerScript : MonoBehaviour
{
    // 2 minutes = 120 seconds (you can change this in Inspector)
    public float timeRemaining = 120f;
    public bool timerIsRunning = true;
    public TMP_Text timerText;

    // Slider + its fill image
    public Slider timerSlider;
    public Image sliderFill;   // drag the "Fill" Image here

    private float totalTime;   // remember starting time

    void Start()
    {
        timeRemaining = 120; // always reset timer
        timerIsRunning = true;

        totalTime = timeRemaining;

        UpdateTimerDisplay(timeRemaining);

        if (timerSlider != null)
        {
            timerSlider.minValue = 0f;
            timerSlider.maxValue = totalTime;
            timerSlider.value = timeRemaining;
        }

        UpdateSliderColor();
    }


    void Update()
    {
        if (!timerIsRunning) return;

        if (timeRemaining > 0f)
        {
            timeRemaining -= Time.deltaTime;
            if (timeRemaining < 0f) timeRemaining = 0f;

            UpdateTimerDisplay(timeRemaining);

            if (timerSlider != null)
            {
                timerSlider.value = timeRemaining;
            }

            UpdateSliderColor();
        }
        else
        {
            timerIsRunning = false;
            TimerFinished();
        }
    }

    void UpdateTimerDisplay(float time)
    {
        int minutes = Mathf.FloorToInt(time / 60f);
        int seconds = Mathf.FloorToInt(time % 60f);
        timerText.text = $"{minutes:00}:{seconds:00}";
    }

    // Green → Yellow → Red as time goes down
    void UpdateSliderColor()
    {
        if (sliderFill == null || totalTime <= 0f) return;

        // 0 = start (full time), 1 = time up
        float t = 1f - (timeRemaining / totalTime);

        Color green = Color.green;
        Color yellow = Color.yellow;
        Color red = Color.red;

        Color c;

        if (t < 0.5f)
        {
            // first half: green → yellow
            float tt = t / 0.5f; // 0..1
            c = Color.Lerp(green, yellow, tt);
        }
        else
        {
            // second half: yellow → red
            float tt = (t - 0.5f) / 0.5f; // 0..1
            c = Color.Lerp(yellow, red, tt);
        }

        sliderFill.color = c;
    }
    public GameObject gameOverPanel;

    void TimerFinished()
    {
        Debug.Log("Time's up!");
        gameOverPanel.SetActive(true);
    }

}
