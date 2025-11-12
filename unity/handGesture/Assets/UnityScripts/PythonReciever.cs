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
                    Debug.Log("Gesture Index: " + data);
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
