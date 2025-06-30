namespace Examples.Tetris
{
    // network op code
    public enum NetOpCode
    {
        ClientAction = 0,
        ServerState = 1,

        GameEvent = 2,

        Feedback = 3,

        ToggleTimestep = 4,
        NextTimestep = 5,

        SpecialModeRequest = 6,
        SpecialModeStepDown = 7,
    }
}