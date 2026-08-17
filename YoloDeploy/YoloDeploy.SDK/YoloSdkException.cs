namespace YoloDeploy.SDK;

public sealed class YoloSdkException : Exception
{
    public YoloSdkException(string message) : base(message)
    {
    }

    public YoloSdkException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
