using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class DamagePopUp : MonoBehaviour
{
    [Header("Text")]
    public TextMeshProUGUI text;

    [Header("Motion")]
    public float lifetime = 0.8f;
    public Vector3 initialVelocity = new Vector3(0f, 1.6f, 0f);
    public float horizontalJitter = 0.25f;
    public float gravity = -2.5f;
    public float scaleOverLife = 0.9f; // 1 = constant, <1 shrinks

    [Header("Fade")]
    public float fadeOutFraction = 0.35f;

    private float _t;
    private Color _baseColor;
    private Vector3 _velocity;

    void Awake()
    {
        if (text == null)
            text = GetComponentInChildren<TextMeshProUGUI>();
    }

    void OnEnable()
    {
        _t = 0f;
        if (text != null)
            _baseColor = text.color;

        // randomize slight horizontal offset and velocity
        var rnd = new Vector2(Random.Range(-horizontalJitter, horizontalJitter), Random.Range(-horizontalJitter, horizontalJitter));
        transform.position += new Vector3(rnd.x, 0f, rnd.y);
        _velocity = initialVelocity + new Vector3(rnd.x, 0f, rnd.y);
    }

    void Update()
    {
        _t += Time.deltaTime;

        // simple rise + gravity
        _velocity.y += gravity * Time.deltaTime;
        transform.position += _velocity * Time.deltaTime;

        // face camera (billboard)
        if (Camera.main != null)
            transform.forward = Camera.main.transform.forward;

        // scale over life
        float life01 = Mathf.Clamp01(_t / lifetime);
        float s = Mathf.Lerp(1f, scaleOverLife, life01);
        transform.localScale = Vector3.one * s;

        // fade out
        float fadeStart = 1f - Mathf.Clamp01(fadeOutFraction);
        float alpha = life01 < fadeStart ? 1f : Mathf.InverseLerp(1f, fadeStart, life01);
        if (text != null)
        {
            var c = _baseColor;
            c.a = alpha;
            text.color = c;
        }

        if (_t >= lifetime)
            Destroy(gameObject);
    }

    public void Init(int amount, Color? color = null)
    {
        if (text != null)
        {
            text.text = $"+{amount}";
            if (color.HasValue)
                text.color = color.Value;
        }
    }

    public static DamagePopUp Spawn(DamagePopUp prefab, Vector3 worldPos, int amount, Color? color = null)
    {
        var inst = Instantiate(prefab, worldPos, Quaternion.identity);
        inst.Init(amount, color);
        return inst;
    }
}
