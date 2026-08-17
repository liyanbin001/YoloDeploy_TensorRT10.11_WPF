namespace YoloDeploy.App;

public sealed class DetectionRow
{
    public int Index { get; init; }
    public string ClassName { get; init; } = "";
    public string ScoreText { get; init; } = "";
    public string BoxText { get; init; } = "";
    public string AngleText { get; init; } = "";
    public string AreaText { get; init; } = "";
}
