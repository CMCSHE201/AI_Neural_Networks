using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;

public class CnnTester : MonoBehaviour
{
    public NNModel modelAsset;
    private Model runtimeCNN;

    public Texture2D testInput;
    public Texture2D testOutput;

    public Image testImage;

    public void Test()
    {
        RenderTexture test = AnalyseImage(testInput);
        testOutput = RenderTexturetoTexture2D(test);
        testImage.sprite = Texture2DtoSprite(testOutput);
    }

    public RenderTexture AnalyseImage(Texture2D texture)
    {
        runtimeCNN = ModelLoader.Load(modelAsset);

        var worker = WorkerFactory.CreateWorker(runtimeCNN);

        Tensor input = new Tensor(texture, 3);

        worker.Execute(input);

        string[] outputNames = runtimeCNN.outputs.ToArray();
        Tensor output = worker.PeekOutput();

        /* Old Image Classification Code
        List<float> temp = output.ToReadOnlyArray().ToList();
        float answer = temp.Max();
        //Debug.LogFormat("Hey that's a {0}", identifiers[temp.IndexOf(answer)]);
        text.text = string.Format("Hey that's a {0}", identifiers[temp.IndexOf(answer)]);
        */

        RenderTexture outputTexture = new RenderTexture(1024, 1024, 3);
        BarracudaTextureUtils.TensorToRenderTexture(output, outputTexture);
        
        input.Dispose();
        output.Dispose();

        return outputTexture;
    }

    public Texture2D RenderTexturetoTexture2D(RenderTexture texture)
    {
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = texture;

        Texture2D finalTexture = new Texture2D(texture.width, texture.height, TextureFormat.RGBA32, false);
        finalTexture.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0, false);
        finalTexture.Apply();

        RenderTexture.active = currentRT;

        return finalTexture;
    }

    public Sprite Texture2DtoSprite(Texture2D texture)
    {
        Sprite sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), new Vector2(0.5f, 0.5f));
        return sprite;
    }

    public Texture2D CopyFromTexture2D(Texture2D texture)
    {
        Texture2D newTexture = new Texture2D(texture.width, texture.height);
        for (int j = 0; j < texture.height; j++)
        {
            for (int i = 0; i < texture.width; i++)
            {
                Color c = texture.GetPixel(i, j);
                newTexture.SetPixel(i, j, c);
            }
        }
        return newTexture;
    }
}