using UnityEngine;
using UnityEngine.UI;

// Simple UI readout for gesture index, speed and computed strength
// Attach to a Canvas GameObject and assign the Text fields in the Inspector.
public class GestureReadout : MonoBehaviour
{
    [Header("Receiver")]
    public pythonreciever pyReciever;

    [Header("UI Text Fields")]
    public Text indexText;
    public Text speedText;
    public Text strengthText;
    public Text lastUpdateText;

    [Header("Strength Computation")]
    [Tooltip("Multiplier to convert px/s into 0..1 strength for display/computing")]
    public float speedToStrength = 1f / 300f;

    void Start()
    {
        if (pyReciever == null)
        {
            pyReciever = FindObjectOfType<pythonreciever>();
            if (pyReciever == null)
                Debug.LogWarning("GestureReadout: pythonreciever not found in scene. Assign it in the inspector.");
            else
                Debug.Log($"GestureReadout auto-assigned pythonreciever from: {pyReciever.gameObject.name}");
        }

        // sanity: ensure UI references are present
        if (indexText == null || speedText == null || strengthText == null || lastUpdateText == null)
            Debug.LogWarning("GestureReadout: One or more UI Text fields are not assigned.");
    }

    void Update()
    {
        if (pyReciever == null)
            return;

        int idx;
        float speed;
        string lastUpdate;
        pyReciever.GetLatest(out idx, out speed, out lastUpdate);

        if (indexText != null)
            indexText.text = $"Index: {idx}";
        if (speedText != null)
            speedText.text = $"Speed: {speed:F1} px/s";

        float strength = Mathf.Clamp01(speed * speedToStrength);
        if (strengthText != null)
            strengthText.text = $"Strength: {strength:F2}";

        if (lastUpdateText != null)
            lastUpdateText.text = string.IsNullOrEmpty(lastUpdate) ? "Last: -" : $"Last: {lastUpdate}";
    }
}
