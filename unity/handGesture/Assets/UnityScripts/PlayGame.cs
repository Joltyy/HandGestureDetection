using UnityEngine;
using UnityEngine.SceneManagement;

public class PlayGame : MonoBehaviour
{
    public void LoadFinalScene()
    {
        SceneManager.LoadScene("final");
    }
}
