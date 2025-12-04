using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class scoreManager : MonoBehaviour
{
    [Header("UI")]
    public TextMeshProUGUI scoreText; // assign a TextMeshPro - Text (UI) in the Inspector

    [Header("Scoring")]
    public int currentScore = 0;
    public float multiplierFactor = 0.2f; // k in formula
    //public float maxMultiplier = 3f;

    // base points per gesture (configurable)
    public int basePunch = 10;
    public int baseSlap = 8;
    public int baseTickle = 5;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        UpdateScoreUI();
    }

    // Update is called once per frame
    void Update()
    {

    }

    public int AddGestureScore(string gestureType, float speed)
    {
        int basePoints = GetBasePoints(gestureType);
        float multiplier = 1f + multiplierFactor * speed;
        //multiplier = Mathf.Clamp(multiplier, 1f, maxMultiplier);

        int added = Mathf.RoundToInt(basePoints * multiplier);
        currentScore += added;

        UpdateScoreUI();
        return added;
    }

    // Lightweight helper to add a small amount of points (e.g., for continuous tickle)
    public void AddPoints(int points)
    {
        if (points <= 0) return;
        currentScore += points;
        UpdateScoreUI();
    }

    void UpdateScoreUI()
    {
        if (scoreText != null)
            scoreText.text = "Score: " + currentScore;
    }

    int GetBasePoints(string gesture)
    {
        switch (gesture.ToLowerInvariant())
        {
            case "punch": return basePunch;
            case "slap": return baseSlap;
            case "tickle": return baseTickle;
            default: return 1;
        }
    }
    void OnGUI()
    {
        if (scoreText == null)
            GUI.Label(new Rect(10, 10, 200, 30), "Score: " + currentScore);
    }
}
