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


    void Start()
    {
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
                    string[] parts = data.Split(',');
                    if (parts.Length >= 1)
                    {
                        int.TryParse(parts[0], out gestureIndex);
                    }
                    if (parts.Length >= 2)
                    {
                        float.TryParse(parts[1], out gestureSpeed);
                    }
                    Debug.Log($"Gesture Index: {gestureIndex}, Speed: {gestureSpeed:F2}");
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
        receiveThread?.Abort();
    }
}
