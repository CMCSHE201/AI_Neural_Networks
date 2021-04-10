using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.Networking;
using UnityEditor.UI;
using UnityEngine.UI;

public class ButtonScript : MonoBehaviour
{

    string path = "";
    public RawImage rawImage;
    public Texture myTexture = null;
    public Text errorText;

    // Start is called before the first frame update
    void Start()
    {
        errorText.gameObject.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void OnButtonClick()
    {
        errorText.gameObject.SetActive(false);

        path = EditorUtility.OpenFilePanel("Show all images (.png) ", "", "png");

        StartCoroutine(DisplayTexture());
    }

    public void OnButtonClick2()
    {
        if (GetTexture() == null)
        {
            errorText.gameObject.SetActive(true);
        }
    }

    IEnumerator DisplayTexture()
    {
        UnityWebRequest www = UnityWebRequestTexture.GetTexture("file:///" + path);

        yield return www.SendWebRequest();

        if (www.isNetworkError || www.isHttpError)
        {
            Debug.Log(www.error);
        }
        else
        {
            myTexture = ((DownloadHandlerTexture)www.downloadHandler).texture;
            rawImage.texture = myTexture;
        }
    }

    public Texture GetTexture()
    {
        return myTexture;
    }
}
