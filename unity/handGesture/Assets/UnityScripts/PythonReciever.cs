using System;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class pythonreciever : MonoBehaviour
{
    private TcpClient client;
    private StreamReader reader;
    private Thread receiveThread;
    private bool running = false;
    private string latestMessage = "";
    public int gestureIndex = 0;
    public float gestureSpeed = 0f;
    public string sourceObjectName = "";
    public string lastUpdateUtc = ""; // ISO timestamp of last received update (for debugging)
    private readonly object dataLock = new object();


    void Start()
    {
        sourceObjectName = gameObject.name;
        Debug.Log($"PythonReciever started on GameObject: {sourceObjectName}");
        ConnectToPython();
    }

    void ConnectToPython()
    {
        try
        {
            client = new TcpClient("127.0.0.1", 5005);
            reader = new StreamReader(client.GetStream());
            running = true;

            receiveThread = new Thread(ReceiveData);
            receiveThread.IsBackground = true;
            receiveThread.Start();

            Debug.Log("Connected to Python server!");
        }
        catch (Exception e)
        {
            Debug.LogError("Connection failed: " + e.Message);
        }
    }

    void ReceiveData()
    {
        try
        {
            while (running)
            {
                string data = reader.ReadLine();
                if (!string.IsNullOrEmpty(data))
                {
                    latestMessage = data;
                    // data format: "gestureIndex,speed"
                    string[] parts = data.Split(',');
                    int parsedIndex = 0;
                    float parsedSpeed = 0f;

                    if (parts.Length >= 1)
                        int.TryParse(parts[0], out parsedIndex);
                    if (parts.Length >= 2)
                        float.TryParse(parts[1], out parsedSpeed);

                    lock (dataLock)
                    {
                        gestureIndex = parsedIndex;
                        gestureSpeed = parsedSpeed;
                        lastUpdateUtc = DateTime.UtcNow.ToString("O");
                    }

                    Debug.Log($"[PythonReciever:{sourceObjectName}] Gesture Index: {parsedIndex}  Speed: {parsedSpeed:F2} px/s  (updated: {lastUpdateUtc})");
                    // values are now stored in the public fields for other scripts to read
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Receive error: " + e.Message);
        }
    }

    void OnApplicationQuit()
    {
        running = false;
        reader?.Close();
        client?.Close();
        try
        {
            if (receiveThread != null && receiveThread.IsAlive)
            {
                receiveThread.Join(100);
            }
        }
        catch (Exception)
        {
            // ignore
        }
    }

    // Thread-safe accessor for other scripts to read the latest data snapshot
    public void GetLatest(out int outGestureIndex, out float outGestureSpeed, out string outLastUpdateUtc)
    {
        lock (dataLock)
        {
            outGestureIndex = gestureIndex;
            outGestureSpeed = gestureSpeed;
            outLastUpdateUtc = lastUpdateUtc;
        }
    }
}
