using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class DamagePopUp : MonoBehaviour
{
    [Header("Text")]
    public TextMeshProUGUI text;

    [Header("Animation")]
    public float lifetime = 0.8f;
    public Vector2 startOffset = Vector2.zero;
    public Vector2 riseOffset = new Vector2(0f, 80f);
    public float horizontalJitter = 12f;
    public AnimationCurve ease = AnimationCurve.EaseInOut(0, 0, 1, 1);

    [Header("Fade")]
    public float fadeOutFraction = 0.35f;

    private RectTransform _rt;
    private float _t;
    private Color _baseColor;
    private Vector2 _spawnAnchoredPos;
    private bool _initialized = false; // guard

    void Awake()
    {
        _rt = GetComponent<RectTransform>();
        if (text == null) text = GetComponentInChildren<TextMeshProUGUI>();
        if (text != null) _baseColor = text.color;
    }

    void OnEnable()
    {
        if (!_initialized) return; // do nothing until Init() called

        _t = 0f;

        float jx = Random.Range(-horizontalJitter, horizontalJitter);
        _spawnAnchoredPos = _rt.anchoredPosition + startOffset + new Vector2(jx, 0f);

        if (text != null)
        {
            var c = text.color;
            c.a = _baseColor.a;
            text.color = c;
        }
    }

    void Update()
    {
        if (!_initialized) return; // skip when not initialized

        _t += Time.deltaTime;
        float life01 = Mathf.Clamp01(_t / lifetime);

        float e = ease.Evaluate(life01);
        _rt.anchoredPosition = _spawnAnchoredPos + riseOffset * e;

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
        _initialized = true; // enable animation

        if (text != null)
        {
            text.text = $"+{amount}";
            if (color.HasValue)
            {
                _baseColor = color.Value;
                text.color = color.Value;
            }
            else
            {
                _baseColor = text.color;
            }
        }

        // force OnEnable setup if already enabled
        OnEnable();
    }

    public static DamagePopUp Spawn(DamagePopUp prefab, Transform parent, Vector2 anchoredPos, int amount, Color? color = null)
    {
        var inst = Instantiate(prefab, parent);
        var rt = inst.GetComponent<RectTransform>();
        rt.anchoredPosition = anchoredPos;
        inst.Init(amount, color);
        return inst;
    }
}
