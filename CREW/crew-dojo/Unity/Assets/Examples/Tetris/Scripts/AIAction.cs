namespace Examples.Tetris
{
    public enum AIAction
    {
        POPULATION_EVALUATION = -1,
        NO_MOVE = 0,
        MOVE_LEFT = 1,
        MOVE_RIGHT = 2,
        ROTATE = 3,
        FLIP = 4,
        DROP = 5,

        Count,

        // used in special mode
        STEP_DOWN = 100,
        SIMULATE = -100,
    }
}