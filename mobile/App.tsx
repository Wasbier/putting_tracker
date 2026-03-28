import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { NavigationContainer } from "@react-navigation/native";
import { StatusBar } from "expo-status-bar";
import React from "react";
import { Text } from "react-native";
import { SafeAreaProvider } from "react-native-safe-area-context";
import HistoryScreen from "./src/screens/HistoryScreen";
import PracticeScreen from "./src/screens/PracticeScreen";
import RecordScreen from "./src/screens/RecordScreen";

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <SafeAreaProvider>
      <NavigationContainer>
        <StatusBar style="light" />
        <Tab.Navigator
          screenOptions={{
            headerShown: false,
            tabBarStyle: {
              backgroundColor: "#062d18",
              borderTopColor: "rgba(244,241,232,0.2)",
            },
            tabBarActiveTintColor: "#f4f1e8",
            tabBarInactiveTintColor: "rgba(244,241,232,0.45)",
            tabBarLabelStyle: { fontSize: 12, fontWeight: "600" },
          }}
        >
          <Tab.Screen
            name="Practice"
            component={PracticeScreen}
            options={{
              tabBarIcon: () => <Text style={{ fontSize: 20 }}>⛳</Text>,
            }}
          />
          <Tab.Screen
            name="History"
            component={HistoryScreen}
            options={{
              tabBarIcon: () => <Text style={{ fontSize: 20 }}>📋</Text>,
            }}
          />
          <Tab.Screen
            name="Record"
            component={RecordScreen}
            options={{
              tabBarIcon: () => <Text style={{ fontSize: 20 }}>🎥</Text>,
            }}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}
