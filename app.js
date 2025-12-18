import React, { useState, useEffect } from 'react';
import { Calendar, Dumbbell, Utensils, Droplets, Moon, Weight, TrendingUp, Clock, CheckCircle, XCircle } from 'lucide-react';

// Data store using React state (no localStorage per instructions)
const FitnessOS = () => {
  const [currentTab, setCurrentTab] = useState('today');
  const [dayData, setDayData] = useState({
    date: new Date().toISOString().split('T')[0],
    dayIndex: 0,
    planned: {
      workout: 'push',
      mealsKcal: 3550,
      hydrationMl: 3200,
      meals: [
        { time: '08:00', kcal: 600, name: 'Breakfast' },
        { time: '11:00', kcal: 550, name: 'Mid-Morning' },
        { time: '14:00', kcal: 700, name: 'Lunch' },
        { time: '17:00', kcal: 550, name: 'Pre-Workout' },
        { time: '20:00', kcal: 700, name: 'Post-Workout' },
        { time: '22:30', kcal: 450, name: 'Dinner' }
      ],
      hydrationWindows: [
        { name: 'Morning', ml: 800, time: '06:00-12:00' },
        { name: 'Afternoon', ml: 800, time: '12:00-17:00' },
        { name: 'Evening', ml: 800, time: '17:00-21:00' },
        { name: 'Night', ml: 800, time: '21:00-23:00' }
      ],
      workoutSets: [
        { exercise: 'Bench Press', sets: 4, reps: 8 },
        { exercise: 'Overhead Press', sets: 3, reps: 10 },
        { exercise: 'Incline DB Press', sets: 3, reps: 12 },
        { exercise: 'Lateral Raises', sets: 3, reps: 15 },
        { exercise: 'Tricep Pushdowns', sets: 3, reps: 12 }
      ]
    },
    logged: {
      weight: null,
      sleepTime: null,
      wakeTime: null,
      meals: [],
      hydration: [],
      workoutSets: []
    },
    derivedMetrics: {
      sleepDurationMin: null,
      hydrationAdherencePct: 0,
      calorieSurplus: 0,
      workoutDurationMin: null
    },
    confidenceFlags: {
      weightLogged: false,
      hydrationComplete: false,
      sleepComplete: false,
      nutritionComplete: false,
      workoutComplete: false
    },
    dataQualityScore: 0
  });

  // Calculate derived metrics whenever logged data changes
  useEffect(() => {
    calculateMetrics();
  }, [dayData.logged]);

  const calculateMetrics = () => {
    const { logged, planned } = dayData;
    const newMetrics = { ...dayData.derivedMetrics };
    const newFlags = { ...dayData.confidenceFlags };

    // Sleep duration
    if (logged.sleepTime && logged.wakeTime) {
      const sleep = new Date(logged.sleepTime);
      const wake = new Date(logged.wakeTime);
      newMetrics.sleepDurationMin = Math.floor((wake - sleep) / 60000);
      newFlags.sleepComplete = true;
    }

    // Hydration adherence
    const totalLogged = logged.hydration.reduce((sum, h) => sum + (h.ml || 0), 0);
    newMetrics.hydrationAdherencePct = Math.round((totalLogged / planned.hydrationMl) * 100);
    newFlags.hydrationComplete = logged.hydration.length === planned.hydrationWindows.length;

    // Calorie surplus
    const totalCalories = logged.meals.reduce((sum, m) => sum + (m.kcal || 0), 0);
    newMetrics.calorieSurplus = totalCalories - planned.mealsKcal;
    newFlags.nutritionComplete = logged.meals.length === planned.meals.length;

    // Weight
    newFlags.weightLogged = logged.weight !== null && logged.weight > 0;

    // Workout
    newFlags.workoutComplete = logged.workoutSets.length === planned.workoutSets.reduce((sum, s) => sum + s.sets, 0);
    
    if (logged.workoutSets.length > 0) {
      const first = new Date(logged.workoutSets[0].timestamp);
      const last = new Date(logged.workoutSets[logged.workoutSets.length - 1].timestamp);
      newMetrics.workoutDurationMin = Math.floor((last - first) / 60000) || 5;
    }

    // Data quality score
    const flags = Object.values(newFlags);
    const score = flags.filter(Boolean).length / flags.length;
    
    setDayData(prev => ({
      ...prev,
      derivedMetrics: newMetrics,
      confidenceFlags: newFlags,
      dataQualityScore: Math.round(score * 100) / 100
    }));
  };

  const logWeight = (weight) => {
    if (weight > 0) {
      setDayData(prev => ({
        ...prev,
        logged: {
          ...prev.logged,
          weight: parseFloat(weight),
          weightTimestamp: new Date().toISOString()
        }
      }));
    }
  };

  const logSleep = () => {
    setDayData(prev => ({
      ...prev,
      logged: {
        ...prev.logged,
        sleepTime: new Date().toISOString()
      }
    }));
  };

  const logWake = () => {
    if (!dayData.logged.sleepTime) {
      alert('Error: Must log sleep time before wake time');
      return;
    }
    setDayData(prev => ({
      ...prev,
      logged: {
        ...prev.logged,
        wakeTime: new Date().toISOString()
      }
    }));
  };

  const logHydration = (windowIndex, ml) => {
    const newHydration = [...dayData.logged.hydration];
    newHydration[windowIndex] = {
      ml: parseInt(ml) || 0,
      loggedAt: ml > 0 ? new Date().toISOString() : null,
      windowName: dayData.planned.hydrationWindows[windowIndex].name
    };
    setDayData(prev => ({
      ...prev,
      logged: {
        ...prev.logged,
        hydration: newHydration
      }
    }));
  };

  const logMeal = (mealIndex, kcal) => {
    const newMeals = [...dayData.logged.meals];
    newMeals[mealIndex] = {
      mealName: dayData.planned.meals[mealIndex].name,
      kcal: parseInt(kcal) || 0,
      loggedAt: kcal > 0 ? new Date().toISOString() : null
    };
    setDayData(prev => ({
      ...prev,
      logged: {
        ...prev.logged,
        meals: newMeals
      }
    }));
  };

  const logWorkoutSet = (exerciseIndex, setNum, actualReps) => {
    const exercise = dayData.planned.workoutSets[exerciseIndex];
    const newSet = {
      exercise: exercise.exercise,
      setNumber: setNum,
      plannedReps: exercise.reps,
      actualReps: parseInt(actualReps) || 0,
      timestamp: new Date().toISOString()
    };
    setDayData(prev => ({
      ...prev,
      logged: {
        ...prev.logged,
        workoutSets: [...prev.logged.workoutSets, newSet]
      }
    }));
  };

  const StatusBadge = ({ complete }) => (
    complete ? 
      <CheckCircle className="w-4 h-4 text-green-500" /> : 
      <XCircle className="w-4 h-4 text-gray-400" />
  );

  const TodayView = () => (
    <div className="space-y-4">
      {/* Header Stats */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-lg font-semibold">Day {dayData.dayIndex}</h2>
          <span className="text-sm text-gray-500">{dayData.date}</span>
        </div>
        <div className="text-2xl font-bold text-blue-600">
          Quality: {(dayData.dataQualityScore * 100).toFixed(0)}%
        </div>
      </div>

      {/* Weight */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Weight className="w-5 h-5 text-purple-600" />
            <h3 className="font-semibold">Weight</h3>
          </div>
          <StatusBadge complete={dayData.confidenceFlags.weightLogged} />
        </div>
        {dayData.logged.weight ? (
          <div className="text-2xl font-bold">{dayData.logged.weight} kg</div>
        ) : (
          <div className="flex gap-2">
            <input
              type="number"
              step="0.1"
              placeholder="Weight (kg)"
              className="flex-1 px-3 py-2 border rounded"
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  logWeight(e.target.value);
                  e.target.value = '';
                }
              }}
            />
          </div>
        )}
      </div>

      {/* Sleep */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Moon className="w-5 h-5 text-indigo-600" />
            <h3 className="font-semibold">Sleep</h3>
          </div>
          <StatusBadge complete={dayData.confidenceFlags.sleepComplete} />
        </div>
        <div className="space-y-2">
          {!dayData.logged.sleepTime ? (
            <button
              onClick={logSleep}
              className="w-full bg-indigo-600 text-white py-2 rounded hover:bg-indigo-700"
            >
              Going to Sleep
            </button>
          ) : !dayData.logged.wakeTime ? (
            <>
              <div className="text-sm text-gray-600">
                Sleep: {new Date(dayData.logged.sleepTime).toLocaleTimeString()}
              </div>
              <button
                onClick={logWake}
                className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700"
              >
                I'm Awake
              </button>
            </>
          ) : (
            <div className="space-y-1">
              <div className="text-sm text-gray-600">
                Sleep: {new Date(dayData.logged.sleepTime).toLocaleTimeString()}
              </div>
              <div className="text-sm text-gray-600">
                Wake: {new Date(dayData.logged.wakeTime).toLocaleTimeString()}
              </div>
              <div className="text-lg font-bold text-green-600">
                Duration: {dayData.derivedMetrics.sleepDurationMin} min ({(dayData.derivedMetrics.sleepDurationMin / 60).toFixed(1)}h)
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Hydration */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Droplets className="w-5 h-5 text-cyan-600" />
            <h3 className="font-semibold">Hydration</h3>
          </div>
          <StatusBadge complete={dayData.confidenceFlags.hydrationComplete} />
        </div>
        <div className="mb-3 text-lg font-bold text-cyan-600">
          {dayData.derivedMetrics.hydrationAdherencePct}% Complete
        </div>
        <div className="space-y-2">
          {dayData.planned.hydrationWindows.map((window, idx) => {
            const logged = dayData.logged.hydration[idx];
            return (
              <div key={idx} className="border rounded p-2">
                <div className="flex justify-between items-center mb-1">
                  <span className="font-medium">{window.name}</span>
                  <span className="text-sm text-gray-500">{window.time}</span>
                </div>
                <div className="text-sm text-gray-600 mb-2">Target: {window.ml} ml</div>
                {logged ? (
                  <div className="text-green-600 font-medium">
                    Logged: {logged.ml} ml
                  </div>
                ) : (
                  <input
                    type="number"
                    placeholder="ml consumed"
                    className="w-full px-3 py-1 border rounded text-sm"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        logHydration(idx, e.target.value);
                        e.target.value = '';
                      }
                    }}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Nutrition */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Utensils className="w-5 h-5 text-orange-600" />
            <h3 className="font-semibold">Nutrition</h3>
          </div>
          <StatusBadge complete={dayData.confidenceFlags.nutritionComplete} />
        </div>
        <div className="mb-3">
          <div className="text-lg font-bold">
            Surplus: <span className={dayData.derivedMetrics.calorieSurplus >= 0 ? 'text-green-600' : 'text-red-600'}>
              {dayData.derivedMetrics.calorieSurplus > 0 ? '+' : ''}{dayData.derivedMetrics.calorieSurplus} kcal
            </span>
          </div>
          <div className="text-sm text-gray-600">
            Target: {dayData.planned.mealsKcal} kcal
          </div>
        </div>
        <div className="space-y-2">
          {dayData.planned.meals.map((meal, idx) => {
            const logged = dayData.logged.meals[idx];
            return (
              <div key={idx} className="border rounded p-2">
                <div className="flex justify-between items-center mb-1">
                  <span className="font-medium">{meal.name}</span>
                  <span className="text-sm text-gray-500">{meal.time}</span>
                </div>
                <div className="text-sm text-gray-600 mb-2">Planned: {meal.kcal} kcal</div>
                {logged ? (
                  <div className="text-green-600 font-medium">
                    Logged: {logged.kcal} kcal
                  </div>
                ) : (
                  <input
                    type="number"
                    placeholder="kcal consumed"
                    className="w-full px-3 py-1 border rounded text-sm"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        logMeal(idx, e.target.value);
                        e.target.value = '';
                      }
                    }}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Workout */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Dumbbell className="w-5 h-5 text-red-600" />
            <h3 className="font-semibold">Workout - {dayData.planned.workout.toUpperCase()}</h3>
          </div>
          <StatusBadge complete={dayData.confidenceFlags.workoutComplete} />
        </div>
        {dayData.derivedMetrics.workoutDurationMin && (
          <div className="mb-3 text-lg font-bold text-red-600">
            Duration: {dayData.derivedMetrics.workoutDurationMin} min
          </div>
        )}
        <div className="space-y-3">
          {dayData.planned.workoutSets.map((exercise, exIdx) => (
            <div key={exIdx} className="border rounded p-3">
              <div className="font-medium mb-2">{exercise.exercise}</div>
              <div className="text-sm text-gray-600 mb-2">
                Planned: {exercise.sets} × {exercise.reps} reps
              </div>
              <div className="space-y-1">
                {Array.from({ length: exercise.sets }).map((_, setIdx) => {
                  const loggedSet = dayData.logged.workoutSets.find(
                    s => s.exercise === exercise.exercise && s.setNumber === setIdx + 1
                  );
                  return (
                    <div key={setIdx} className="flex items-center gap-2">
                      <span className="text-sm w-12">Set {setIdx + 1}:</span>
                      {loggedSet ? (
                        <span className="text-green-600 font-medium">
                          {loggedSet.actualReps} reps
                        </span>
                      ) : (
                        <input
                          type="number"
                          placeholder="reps"
                          className="flex-1 px-2 py-1 border rounded text-sm"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              logWorkoutSet(exIdx, setIdx + 1, e.target.value);
                              e.target.value = '';
                            }
                          }}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const AnalysisView = () => (
    <div className="space-y-4">
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="font-semibold mb-4">Today's Analysis</h3>
        <div className="space-y-3">
          <div className="flex justify-between items-center pb-2 border-b">
            <span>Data Quality Score</span>
            <span className="font-bold text-blue-600">{(dayData.dataQualityScore * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between items-center pb-2 border-b">
            <span>Weight Logged</span>
            <StatusBadge complete={dayData.confidenceFlags.weightLogged} />
          </div>
          <div className="flex justify-between items-center pb-2 border-b">
            <span>Sleep Complete</span>
            <StatusBadge complete={dayData.confidenceFlags.sleepComplete} />
          </div>
          <div className="flex justify-between items-center pb-2 border-b">
            <span>Hydration Complete</span>
            <StatusBadge complete={dayData.confidenceFlags.hydrationComplete} />
          </div>
          <div className="flex justify-between items-center pb-2 border-b">
            <span>Nutrition Complete</span>
            <StatusBadge complete={dayData.confidenceFlags.nutritionComplete} />
          </div>
          <div className="flex justify-between items-center pb-2 border-b">
            <span>Workout Complete</span>
            <StatusBadge complete={dayData.confidenceFlags.workoutComplete} />
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="font-semibold mb-4">Metrics Summary</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Sleep Duration</span>
            <span className="font-medium">
              {dayData.derivedMetrics.sleepDurationMin ? 
                `${(dayData.derivedMetrics.sleepDurationMin / 60).toFixed(1)}h` : 
                'Not logged'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Hydration</span>
            <span className="font-medium">{dayData.derivedMetrics.hydrationAdherencePct}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Calorie Surplus</span>
            <span className={`font-medium ${dayData.derivedMetrics.calorieSurplus >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {dayData.derivedMetrics.calorieSurplus > 0 ? '+' : ''}{dayData.derivedMetrics.calorieSurplus}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Workout Duration</span>
            <span className="font-medium">
              {dayData.derivedMetrics.workoutDurationMin ? 
                `${dayData.derivedMetrics.workoutDurationMin} min` : 
                'Not started'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-gray-900">Fitness OS</h1>
          <p className="text-sm text-gray-500">12-Week Bulk Program</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white border-b sticky top-[72px] z-10">
        <div className="max-w-4xl mx-auto px-4">
          <div className="flex gap-1">
            {['today', 'analysis'].map((tab) => (
              <button
                key={tab}
                onClick={() => setCurrentTab(tab)}
                className={`px-6 py-3 font-medium capitalize transition-colors ${
                  currentTab === tab
                    ? 'text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-4xl mx-auto px-4 py-6 pb-24">
        {currentTab === 'today' && <TodayView />}
        {currentTab === 'analysis' && <AnalysisView />}
      </div>
    </div>
  );
};

export default FitnessOS;
