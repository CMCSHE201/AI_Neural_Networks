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
    public RawImage rawResultingImage;
    public Texture2D myTexture = null;
    public Texture2D myResultingTexture = null;
    public Text errorText;

    public int resWidth = 2550;
    public int resHeight = 3300;

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
        else
        {
            ExportPicture(myResultingTexture);
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
            myResultingTexture = ((DownloadHandlerTexture)www.downloadHandler).texture;
            rawImage.texture = myTexture;
            rawResultingImage.texture = myResultingTexture;
        }
    }

    public Texture GetTexture()
    {
        return myTexture;
    }

    public void ExportPicture(Texture2D t)
    {
        Texture2D exTexture = new Texture2D(t.width, t.height, TextureFormat.RGBA32, false);
        exTexture.SetPixels(t.GetPixels());
        string fileName = TextureName(resWidth, resHeight);
        //string exportName = "Results" + "\\" + fileName;
        System.IO.File.WriteAllBytes(fileName, exTexture.EncodeToPNG());
    }

    public static string TextureName(int width, int height)
    {
        return string.Format("{0}/Results/texture_{1}x{2}_{3}.png",
                             Application.dataPath,
                             width, height,
                             System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
    }
}
